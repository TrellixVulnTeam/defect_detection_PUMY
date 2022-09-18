import argparse
import glob
import json
import multiprocessing
import os.path
import numpy
from tqdm import tqdm

import cv2


def iou(box, clusters):
    x = numpy.minimum(clusters[:, 0], box[0])
    y = numpy.minimum(clusters[:, 1], box[1])
    if numpy.count_nonzero(x == 0) > 0 or numpy.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def avg_iou(boxes, clusters):
    return numpy.mean([numpy.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def k_means(boxes, k, dist=numpy.median, max_iter=300):
    rows = boxes.shape[0]
    distances = numpy.empty((rows, k))
    last_clusters = numpy.zeros((rows,))
    numpy.random.seed()

    clusters = boxes[numpy.random.choice(rows, k, replace=False)]
    iter_count = 0
    while True:
        print('current iter: {}'.format(str(iter_count)))
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        nearest_clusters = numpy.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters
        if iter_count == max_iter:
            break
        iter_count += 1

    return clusters


class AnchorPropose:
    def __init__(self, work_dir, output_path, process_num=8):
        self.work_dir = work_dir
        self.output_path = output_path
        self.process_num = process_num
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def get_bbox_list(self):
        source_json_list = glob.glob(os.path.join(self.work_dir, '*.json'))
        bbox_list = multiprocessing.Manager().list()
        if source_json_list:
            cv2.setNumThreads(0)
            process_pool = multiprocessing.Pool(processes=self.process_num)
            pbar = tqdm(total=len(source_json_list))
            for json_file in source_json_list:
                process_pool.apply_async(self.add_bbox, args=(json_file, bbox_list), callback=lambda _: pbar.update())
            process_pool.close()
            process_pool.join()
        return numpy.array(bbox_list)

    @staticmethod
    def add_bbox(json_file, bbox_list):
        try:
            with open(json_file) as input_file:
                instance_item = json.load(input_file)
            for shape_item in instance_item['shapes']:
                points = numpy.array(shape_item['points'])
                if shape_item['shape_type'] == 'circle':
                    radius = numpy.linalg.norm(points[0] - points[1])
                    bbox_with, bbox_height = 2 * radius, 2 * radius
                else:
                    points_x, points_y = points.T
                    min_x, min_y = min(points_x), min(points_y)
                    max_x, max_y = max(points_x), max(points_y)
                    bbox_with, bbox_height = max_x - min_x, max_y - min_y

                if bbox_with == 0 or bbox_height == 0:
                    continue
                bbox_list.append([bbox_with, bbox_height])
        except:
            pass


def get_parser():
    parser = argparse.ArgumentParser(description='The tool used to plot the bbox distribution.')
    parser.add_argument('-w', '--work_dir', required=True, type=str, default=None, help='The path of the work dictionary.')
    parser.add_argument('-o', '--output_path', type=str, help='The output path of result.')
    parser.add_argument('-c', '--cluster_num', type=int, help='The cluster num.')
    parser.add_argument('-p', '--process_num', type=int, default=8, help='The num of workers for multiprocess. (default: 8)')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = get_parser()
    propose_worker = AnchorPropose(args.work_dir, args.output_path, args.process_num)
    bboxes = propose_worker.get_bbox_list()

    out = k_means(bboxes, k=args.cluster_num)
    print('Boxes:')
    print(out)
    print("Accuracy: {:.2f}%".format(avg_iou(bboxes, out) * 100))
    final_anchors = numpy.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Before Sort Ratios:\n {}".format(final_anchors))
    print("After Sort Ratios:\n {}".format(sorted(final_anchors)))
    index = numpy.argsort(out[:, 0] * out[:, 1])
    print(out[index])
