import os
import traceback
import functools
from collections import OrderedDict
import signal
import urllib.request
import ssl
import queue
import threading
import multiprocessing
import multiprocessing.pool
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from torch import distributed as dist
from torch._utils import _flatten_dense_tensors, _take_tensors, _unflatten_dense_tensors

from .misc_util import get_dist_info


def init_dist(launcher, backend="nccl", **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    if launcher == "pytorch":
        _init_dist_pytorch(backend, **kwargs)
    else:
        raise ValueError(f"Invalid launcher type: {launcher}")


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ["RANK"])
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
    else:
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, **kwargs)


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def allreduce_params(params, coalesce=True, bucket_size_mb=-1):
    """Allreduce parameters.
    Args:
        params (list[torch.Parameters]): List of parameters or buffers of a
            model.
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    params = [param.data for param in params]
    if coalesce:
        _allreduce_coalesced(params, world_size, bucket_size_mb)
    else:
        for tensor in params:
            dist.all_reduce(tensor.div_(world_size))


def allreduce_grads(params, coalesce=True, bucket_size_mb=-1):
    """Allreduce gradients.
    Args:
        params (list[torch.Parameters]): List of parameters of a model
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    grads = [
        param.grad.data
        for param in params
        if param.requires_grad and param.grad is not None
    ]
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))


def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
            bucket, _unflatten_dense_tensors(flat_tensors, bucket)
        ):
            tensor.copy_(synced)


def reduce_mean(tensor):
    """ "Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# Reference: https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class CustomMpPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def init_worker(id_queue=None, param=None):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if id_queue:
        if param is None:
            _PARALLEL_UTIL_FUNC_INIT(id_queue.get())
        else:
            _PARALLEL_UTIL_FUNC_INIT(id_queue.get(), param)


_PARALLEL_UTIL_FUNC_INIT = None


def multi_process(
    msgs,
    func,
    worker=None,
    total=None,
    ret_func=None,
    func_init=None,
    func_init_param=None,
):
    """Multi process util.

    Args:
        msgs (ist or iterable): if msgs is a list, tqdm.total = len(msgs),
            otherwise use the specified 'total' from param.
        func (function): worker function. Msg from msgs will be passed to it.
        worker (int, optional): if not proviced, the worker num will be set as
            the max cpu_count of the machine.
        total (int, optional): if msgs is an iterator, 'total' will be passed
            to tqdm in order to show a complete progress bar.
        ret_func (funciton, optional): if some processed data need to be
            handled in the main process, return it from func and receive it
            from ret_func's input param.
        func_init (function, optional): if func_init_param (list or tuple or
            anything, optional): any param.

    Example:

        class Classifier:
            def __init__(self, checkpoint_path, gpu_id):
                self.model = Model(checkpoint_path, gpu_id)

            def predict(self, data):
                score = self.model.forward(data)
                return score

        def func_init(wid, param):
            global worker_index
            global gpu_id
            global classifier

            worker_index = wid
            gpu_id = worker_index % 8
            checkpoint_path = msg
            classidier = Classifier(checkpoint_path, gpu_id)

        def func(msg):
            global worker_index
            global gpu_id
            global classifier

            rowkey, cover, title = msg
            # preprocessing
            data = ...
            score = classifier.predict(data)

            # return is not necessary
            return rowkey, score

        def ret_func(msg):
            rowkey, score = msg
            global all_predict_results
            all_predict_results[rowkey] = score

        all_predict_results = dict()
        msgs = [line.strip().split() for line in open('data.txt', 'r')]
        rowkeys = [msg[0] for msg in msgs]

        multi_process(
            msgs,
            func,
            worker=16,
            ret_func=ret_func,
            func_init=func_init,
            func_init_param=(checkpoint_path,))
        with open('results.txt', 'w') as f:
            for rowkey in rowkeys:
                f.write('{} {}\n'.format(rowkey, all_predict_results[rowkey]))

    """
    if worker is None:
        cores = multiprocessing.cpu_count()
        worker = cores
    if func_init:
        global _PARALLEL_UTIL_FUNC_INIT
        _PARALLEL_UTIL_FUNC_INIT = func_init
        manager = multiprocessing.Manager()
        id_queue = manager.Queue()
        for i in range(worker):
            id_queue.put(i)
        pool = CustomMpPool(worker, init_worker, (id_queue, func_init_param))
    else:
        pool = CustomMpPool(worker, init_worker)
    if type(msgs) is list:
        total_cnt = len(msgs)
    else:
        total_cnt = total
    try:
        with tqdm(total=total_cnt) as pbar:
            for ret in pool.imap_unordered(func, msgs):
                pbar.update(1)
                if ret_func is not None:
                    ret_func(ret)

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
    finally:
        pool.terminate()
        pool.join()


def fetch_image_func(curlists):
    while True:
        image_info = curlists.get_nowait()  # read with no wait
        i = curlists.qsize()
        print("remain %d task" % i)

        if os.path.exists(image_info[0]):
            print("File have already exist. skip")
        else:
            try:
                # download without validation
                ssl._create_default_https_context = ssl._create_unverified_context

                urllib.request.urlretrieve(image_info[1], filename=image_info[0])
            except OSError:
                print("---------------------------------------------------------")
                print("Exeception information:")
                print(traceback.format_exc())
                print("URL Path: " + image_info[1])
                print("---------------------------------------------------------")


def multi_thread_downloads(curlists):
    num = 10
    threads = []
    for i in range(num):
        t = threading.Thread(
            target=fetch_image_func, args=(curlists,), name="child_thread_%d" % i
        )
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def get_pic_by_url_list(download_save_path, name_list, url_list, max_num=2000):
    curlists = queue.Queue()

    for j, url_path in enumerate(url_list):
        if j % max_num == 0:
            sub_dir = str(j // max_num + 1)
            sub_path = os.path.join(download_save_path, sub_dir)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
        file_path = os.path.join(sub_path, name_list[j])
        curlists.put((file_path, url_path))

    multi_thread_downloads(curlists)


def get_pic_by_url(download_save_path, anno_path, max_num=2000):
    for txt_file in os.listdir(anno_path):
        if not txt_file.endswith("txt"):
            continue
        download_path = os.path.join(download_save_path, txt_file[:-4])
        if not os.path.exists(download_path):
            print("Download folder not exist, try to create it.")
            os.makedirs(download_path)

        print("Try downloading pics in the file: {}".format(txt_file))

        url_list = open(os.path.join(anno_path, txt_file), "r").readlines()

        curlists = queue.Queue()

        for j, url_file in enumerate(url_list):
            if j % max_num == 0:
                sub_dir = str(j // max_num + 1)
                sub_path = os.path.join(download_path, sub_dir)
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)
                print("Dealing with: ", sub_path)
            url_path = url_file.strip()
            url_id = url_file.split("/")[-2]
            file_path = os.path.join(sub_path, url_id + ".jpg")
            curlists.put((file_path, url_path))

        multi_thread_downloads(curlists)
