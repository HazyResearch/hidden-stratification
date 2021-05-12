import ast
from collections import MutableMapping
import datetime
from datetime import datetime, timedelta
from functools import singledispatch
import json
import random
import subprocess
import time
import uuid

import numpy as np
import torch

tenmin_td = timedelta(minutes=10)
hour_td = timedelta(hours=1)


def format_timedelta(timedelta):
    s = str(timedelta)
    if timedelta < tenmin_td:
        return s[3:]
    if timedelta < hour_td:
        return s[2:]
    return s


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ScientificNotationDecoder(json.JSONDecoder):
    """Decodes floats incorrectly parsed by ActionJsonSchema (e.g. 1e-5)"""

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        for k, v in obj.items():
            if type(v) == str:
                obj[k] = convert_value(v)
        return obj


@singledispatch
def keys_to_strings(ob):
    """
    Converts keys in a dictionary object to strings for JSON.

    source:
    https://stackoverflow.com/questions/47568356/python-convert-all-keys-to-strings
    """
    if type(ob) == dict:
        return {str(k): keys_to_strings(v) for k, v in ob.items()}
    elif type(ob) == list:
        return [keys_to_strings(v) for v in ob]
    return ob


def convert_value(value):
    """Parse string as python literal if possible and fallback to string."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def get_unique_str():
    date = datetime.now().strftime('%Y-%m-%d')
    time = datetime.now().strftime('%H-%M-%S')
    rand = str(uuid.uuid4())[:8]
    return f'{date}_{time}_{rand}'


def set_by_dotted_path(d, path, value):
    """
    Change an entry in a nested dict using a dotted path.
    Raises exception if path not in d.

    Examples
    --------
    >>> d = {'foo': {'bar': 7}}
    >>> set_by_dotted_path(d, 'foo.bar', 10)
    >>> d
    {'foo': {'bar': 10}}
    >>> set_by_dotted_path(d, 'foo.d.baz', 3)
    >>> d
    {'foo': {'bar': 10, 'd': {'baz': 3}}}
    """
    split_path = path.split(".")
    split_path_len = len(split_path)

    current_option = d
    for idx, p in enumerate(split_path):
        assert p in current_option, f'Path {split_path} does not exist in dictionary.'

        if idx != split_path_len - 1:
            current_option = current_option[p]
        else:
            current_option[p] = value


def merge_dicts(a, b):
    """
    Returns a dictionary in which b is merged into a.

    This is different than the naive approach {**a, **b} because it preserves
    all existing values in nested dictionaries that appear in both a and b,
    rather than overwriting a's entire nested dictionary with b's.
    """

    def merge_dicts_rec(a, b):
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    merge_dicts_rec(a[key], b[key])
                else:
                    a[key] = b[key]  # overwrite values in a
            else:
                a[key] = b[key]
        return a

    return merge_dicts_rec(dict(a), b)


def get_git_commit_info():
    get_commit_hash = "git log | head -n 1 | awk '{print $2}'"
    check_unstaged = 'git diff --exit-code'
    check_staged = 'git diff --cached --exit-code'
    status = 'git status'
    cmds = [get_commit_hash, check_unstaged, check_staged, status]
    do_checks = [True, False, False, True]
    saved_infos = []
    for cmd, do_check in zip(cmds, do_checks):
        try:
            process_result = subprocess.run(
                cmd,
                shell=True,
                check=do_check,
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            saved_infos.append((process_result.returncode, process_result.stdout.strip()))
            err_msg = None
        except subprocess.CalledProcessError as e:
            err_msg = str(e)
            returncode = int(err_msg.split()[-1][:-1])
    if err_msg is not None:
        return err_msg
    commit_hash = saved_infos[0][1]
    msg = 'Current commit: ' + commit_hash
    if saved_infos[1][0] or saved_infos[2][0]:
        msg += '; Uncommitted changes present'
    return msg


def set_seed(seed, use_cuda):
    if seed is None or seed < 0:
        random.seed(time.perf_counter())
        seed = random.randint(0, 100000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def init_cuda(deterministic, allow_multigpu=False):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    if torch.cuda.device_count() > 1 and not allow_multigpu:
        raise RuntimeError('Multi-GPU training unsupported. Run with CUDA_VISIBLE_DEVICES=X')
    return use_cuda


def move_to_device(obj, device):
    r"""Given a structure (possibly) containing Tensors on the CPU, move all the Tensors
      to the specified GPU (or do nothing, if they should be on the CPU).

      Adapted from https://github.com/SenWu/emmental/blob/master/src/emmental/utils/utils.py

        device = -1 -> "cpu"
        device =  0 -> "cuda:0"

      Originally from:
        https://github.com/HazyResearch/metal/blob/mmtl_clean/metal/utils.py

    Args:
      obj(Any): The object to convert.
      device(int): The device id, defaults to -1.

    Returns:
      Any: The converted object.
    """

    if device < 0 or not torch.cuda.is_available():
        return obj.cpu()
    elif isinstance(obj, torch.Tensor):
        return obj.cuda(device)  # type: ignore
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(item, device) for item in obj])
    else:
        return obj


def save_config(config, save_path):
    f = open(save_path, 'w')
    json_str = json.dumps(json.loads(jsonpickle.encode(config)), indent=4)
    f.write(json_str)
    f.close()


def load_config(load_path):
    f = open(load_path, 'r')
    config = jsonpickle.decode(f.read())
    f.close()
    return config


def flatten_dict(d, parent_key='', sep='_'):
    '''
    Source: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    '''
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def concatenate_iterable(list_of_iterables):
    if isinstance(list_of_iterables[0], torch.Tensor):
        return torch.cat([x.detach().cpu() for x in list_of_iterables]).numpy()
    elif isinstance(list_of_iterables[0], np.ndarray):
        return np.concatenate(list_of_iterables)
    elif isinstance(list_of_iterables[0], list):
        return np.array(list(itertools.chain(*list_of_iterables)))


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        if 'lr' in param_group:
            return param_group['lr']
