import os.path as osp
import pickle
import shutil
import tempfile
import time
import torch
import torch.distributed as dist
import numpy as np

from mtl.utils.misc_util import ProgressBar
from mtl.utils.photometric_util import tensor2imgs
from mtl.utils.misc_util import get_dist_info
from mtl.utils.mask_util import encode_mask_results
from mtl.utils.io_util import file_load, obj_dump
from mtl.utils.geometric_util import imresize
from mtl.utils.path_util import mkdir_or_exist


def single_device_det_test(
    model, data_loader, show=False, out_dir=None, show_score_thr=0.3
):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    for _, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            img_tensor = data["img"][0]
            img_metas = data["img_metas"]
            imgs = tensor2imgs(img_tensor, **img_metas[0][0]["img_norm_cfg"])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta[0]["img_shape"]
                img_show = img[:h, :w, [2, 1, 0]]

                ori_h, ori_w = img_meta[0]["ori_shape"][:-1]
                img_show = imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta[0]["file_name"])
                else:
                    out_file = None

                if not (out_file.endswith("jpg") or out_file.endswith("png")):
                    out_file = out_file + ".jpg"

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr,
                )

        # encode mask results
        if isinstance(result[0], tuple):
            result = [
                (bbox_results, encode_mask_results(mask_results))
                for bbox_results, mask_results in result
            ]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_device_det_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [
                    (bbox_results, encode_mask_results(mask_results))
                    for bbox_results, mask_results in result
                ]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def single_device_cls_test(model, data_loader, show, out_dir, save_txt_path=None):
    model.eval()
    results = []
    result_names = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    for _, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            img_tensor = data["img"]
            img_metas = data["img_metas"]
            imgs = tensor2imgs(img_tensor, **img_metas[0]["img_norm_cfg"])
            assert len(imgs) == len(img_metas)

            for j, img in enumerate(imgs):
                h, w, _ = img_metas["img_shape"][j]
                img_show = img[:h, :w, [2, 1, 0]]

                ori_h = img_metas["ori_shape"][j][0]
                ori_w = img_metas["ori_shape"][j][1]

                img_show = imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_metas["file_name"][j] + ".jpg")
                else:
                    out_file = None

                label_id = np.argmax(result[j])
                label_name = dataset.class_names[label_id]
                label_score = result[j][label_id]
                model.module.show_result(
                    img_show, {label_name: label_score}, show=show, out_file=out_file
                )

        results.extend(result)
        if save_txt_path is not None and save_txt_path != "":
            img_metas = data["img_metas"]
            for file_name in img_metas["file_name"]:
                result_names.append(file_name)

        for _ in range(batch_size):
            prog_bar.update()

    if save_txt_path is not None and save_txt_path != "":
        fw = open(save_txt_path, "w")
        for (res_name, res_value) in zip(result_names, results):
            tmp_str = res_name
            for v_item in res_value:
                tmp_str += ", " + str(v_item)
            tmp_str += "\n"
            fw.write(tmp_str)
        fw.close()

    return results


def multi_device_cls_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = ProgressBar(len(dataset))
    time.sleep(2)
    for _, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(rescale=True, **data)

        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def single_device_seg_test(model, data_loader, show, out_dir):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    for _, data in enumerate(data_loader):

        with torch.no_grad():
            result = model(**data)

        batch_size = len(result)
        if show or out_dir:
            img_tensor = data["img"][0]

            img_metas = data["img_metas"]
            imgs = tensor2imgs(img_tensor, **img_metas[0][0]["img_norm_cfg"])
            assert len(imgs) == len(img_metas)

            for _, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta[0]["img_shape"]
                img_show = img[:h, :w, [0, 1, 2]]
                # img_show = img[:h, :w, [2, 1, 0]]

                ori_h = img_meta[0]["ori_shape"][0]
                ori_w = img_meta[0]["ori_shape"][1]
                img_show = imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    if img_meta[0]["file_name"].endswith("jpg"):
                        out_file = osp.join(out_dir, img_meta[0]["file_name"])
                    else:
                        out_file = osp.join(out_dir, img_meta[0]["file_name"] + ".jpg")
                else:
                    out_file = None

                model.module.class_names = dataset.class_names
                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.palette,
                    show=show,
                    out_file=out_file,
                )

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()

    return results


def multi_device_seg_test(model, data_loader, out_dir, gpu_collect):
    pass


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device="cuda")
        if rank == 0:
            mkdir_or_exist(".dist_test")
            tmpdir = tempfile.mkdtemp(dir=".dist_test")
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device="cuda"
            )
            dir_tensor[: len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    obj_dump(result_part, osp.join(tmpdir, f"part_{rank}.pkl"))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f"part_{i}.pkl")
            part_list.append(file_load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device="cuda"
    )
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device="cuda")
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device="cuda")
    part_send[: shape_tensor[0]] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(pickle.loads(recv[: shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


def single_device_test(
    model,
    data_loader,
    model_type="det",
    show=False,
    out_dir=None,
    show_score_thr=0.3,
    save_txt_path=None,
):

    if model_type == "det":
        return single_device_det_test(model, data_loader, show, out_dir, show_score_thr)
    elif model_type == "cls":
        return single_device_cls_test(model, data_loader, show, out_dir, save_txt_path)
    elif model_type == "seg":
        return single_device_seg_test(model, data_loader, show, out_dir)
    else:
        return None


def multi_device_test(
    model, data_loader, model_type="det", tmpdir=None, gpu_collect=False
):
    if model_type == "det":
        return multi_device_det_test(model, data_loader, tmpdir, gpu_collect)
    elif model_type == "cls":
        return multi_device_cls_test(model, data_loader, tmpdir, gpu_collect)
    elif model_type == "seg":
        return multi_device_seg_test(model, data_loader, tmpdir, gpu_collect)
    else:
        return None
