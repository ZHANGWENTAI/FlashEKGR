# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import os.path as osp
import queue
import random
import time
import traceback
import zipfile
from collections import defaultdict
from functools import wraps
from _thread import start_new_thread
from urllib import request as ur

import numpy as np
import torch
from tqdm import tqdm

GBFACTOR = float(1 << 30)


def cal_ent_loc(query_structure, idx):
    if query_structure[0] == "<":
        return cal_ent_loc(query_structure[1], idx)
    
    # Check if this is an all-relation path query
    all_relation_flag = True
    if not isinstance(query_structure, tuple) or len(query_structure) != 2:
        all_relation_flag = False
    elif query_structure[0] != "e":
        all_relation_flag = False
    elif not isinstance(query_structure[-1], tuple):
        all_relation_flag = False
    else:
        for ele in query_structure[-1]:
            if ele not in ["r", "n"]:
                all_relation_flag = False
                break
    
    ent_locations = []
    if all_relation_flag:
        if query_structure[0] == "e":
            ent_locations.append(idx)
            idx += 1
        else:
            ent_locations, idx = cal_ent_loc(query_structure[0], idx)
        for i in range(len(query_structure[-1])):
            idx += 1
    else:
        # Check if the last element is a simple tail (contains only "r", "n", "i", "u")
        last_elem = query_structure[-1]
        if isinstance(last_elem, tuple) and all(isinstance(ele, str) and ele in ["r", "n", "i", "u"] for ele in last_elem):
            # Standard structure: branches + tail
            for i in range(len(query_structure) - 1):
                tmp_ent_locations, idx = cal_ent_loc(query_structure[i], idx)
                ent_locations.extend(tmp_ent_locations)
            # Tail doesn't add entity locations, just relation indices
            for ele in last_elem:
                if ele in ["r", "n"]:
                    idx += 1
        else:
            # All elements are branches (e.g., for "2i", "3i" queries)
            for i in range(len(query_structure)):
                if query_structure[i] == "u":
                    assert i == len(query_structure) - 1
                    break
                tmp_ent_locations, idx = cal_ent_loc(query_structure[i], idx)
                ent_locations.extend(tmp_ent_locations)
    return ent_locations, idx


def cal_ent_loc_dict(query_name_dict):
    query_ent_loc_dict = {}
    for query_structure in query_name_dict:
        # if query_name_dict[query_structure] == '2u-DNF':
        #     tmp_structure = ('e', ('r',))
        # elif query_name_dict[query_structure] == 'up-DNF':
        #     tmp_structure = ('e', ('r', 'r'))
        # else:
        #     tmp_structure = query_structure
        query_ent_loc_dict[query_structure] = cal_ent_loc(query_structure, 0)[0]
    return query_ent_loc_dict


def list2tuple(l):
    return tuple(list2tuple(x) if isinstance(x, list) else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if isinstance(x, tuple) else x for x in t)


def flatten(l):
    return sum(map(flatten, l), []) if isinstance(l, tuple) else [l]


def flatten_list(l):
    return sum(map(flatten_list, l), []) if isinstance(l, list) else [l]


def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if isinstance(arg_return, tuple):
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except BaseException:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return


def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries


def construct_graph(base_path, indexified_files):
    # knowledge graph
    # kb[e][rel] = set([e, e, e])
    ent_in, ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    for indexified_p in indexified_files:
        with open(osp.join(base_path, indexified_p)) as f:
            for i, line in enumerate(f):
                if len(line) == 0:
                    continue
                e1, rel, e2 = line.split("\t")
                e1 = int(e1.strip())
                e2 = int(e2.strip())
                rel = int(rel.strip())
                ent_out[e1][rel].add(e2)
                ent_in[e2][rel].add(e1)

    return ent_in, ent_out


def tuple2filterlist(t):
    return list(tuple2filterlist(x) if isinstance(x, tuple) else -1 if x == "u" else -2 if x == "n" else x for x in t)


def achieve_answer_with_constraints(query, ent_in, ent_out, max_to_keep):
    assert isinstance(query[-1], list)
    all_relation_flag = True
    for ele in query[-1]:
        if (not isinstance(ele, int)) or (ele == -1):
            all_relation_flag = False
            break
    if all_relation_flag:
        if isinstance(query[0], int):
            ent_set = set([query[0]])
        else:
            ent_set = achieve_answer_with_constraints(query[0], ent_in, ent_out, max_to_keep)
        for i in range(len(query[-1])):
            if query[-1][i] == -2:
                assert False, "negation not supported"
                ent_set = set(range(len(ent_in))) - ent_set
            else:
                ent_set_traverse = set()
                n_traversed = 0
                for idx, ent in enumerate(ent_set):
                    if n_traversed == max_to_keep:
                        break
                    if query[-1][i] in ent_out[ent]:
                        ent_set_traverse = ent_set_traverse.union(ent_out[ent][query[-1][i]])
                        n_traversed += 1
                ent_set = ent_set_traverse
    else:
        ent_set = achieve_answer_with_constraints(query[0], ent_in, ent_out, max_to_keep)
        union_flag = False
        if len(query[-1]) == 1 and query[-1][0] == -1:
            assert False, "union not supported"
            union_flag = True
        for i in range(1, len(query)):
            if not union_flag:
                ent_set = ent_set.intersection(achieve_answer_with_constraints(query[i], ent_in, ent_out, max_to_keep))
            else:
                if i == len(query) - 1:
                    continue
                ent_set = ent_set.union(achieve_answer_with_constraints(query[i], ent_in, ent_out, max_to_keep))
    return ent_set


def fill_query(query_structure, ent_in, ent_out, answer, chill=False):
    assert isinstance(query_structure[-1], list)
    all_relation_flag = True
    for ele in query_structure[-1]:
        if ele not in ["r", "n"]:
            all_relation_flag = False
            break
    if all_relation_flag:
        r = -1
        for i in range(len(query_structure[-1]))[::-1]:
            if query_structure[-1][i] == "n":
                query_structure[-1][i] = -2
                continue
            if chill:
                r = random.sample(ent_in[answer].keys(), 1)[0]
            else:
                found = False
                for j in range(40):
                    r_tmp = random.sample(ent_in[answer].keys(), 1)[0]
                    if r_tmp // 2 != r // 2 or r_tmp == r:
                        r = r_tmp
                        found = True
                        break
                if not found:
                    return True
            query_structure[-1][i] = r
            answer = random.sample(ent_in[answer][r], 1)[0]
            # elif query_structure[-1][i] == 'n':
            #     assert False
            # answer = random.sample(set(range(len(ent_in))) - answer, 1)[0]
            # #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!make sure this negative entity is of
            # same type with the current asnwer.
        if query_structure[0] == "e":
            query_structure[0] = answer
        else:
            return fill_query(query_structure[0], ent_in, ent_out, answer, chill)
    else:
        same_structure = defaultdict(list)
        for i in range(len(query_structure)):
            same_structure[list2tuple(query_structure[i])].append(i)
        for i in range(len(query_structure)):
            if len(query_structure[i]) == 1 and query_structure[i][0] == "u":
                assert i == len(query_structure) - 1
                query_structure[i][0] = -1
                continue
            broken_flag = fill_query(query_structure[i], ent_in, ent_out, answer, chill)
            if broken_flag:
                return True
        if not chill:
            for structure in same_structure:
                if len(same_structure[structure]) != 1:
                    structure_set = set()
                    for i in same_structure[structure]:
                        structure_set.add(list2tuple(query_structure[i]))
                    if len(structure_set) < len(same_structure[structure]):
                        # print('same query')
                        return True
        return False


def sample_negative_bidirectional(query, ent_in, ent_out, nent):  # pylint: disable=unused-argument
    """Sample negative examples bidirectionally (placeholder)."""
    # Placeholder function - implementation needed


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition("/")[2]
    path = osp.join(folder, filename)

    if osp.exists(path) and osp.getsize(path) > 0:  # pragma: no cover
        if log:
            print("Using exist file", filename)
        return path

    if log:
        print("Downloading", url)

    if not osp.exists(folder):
        os.makedirs(folder)

    chunk_size = 1024 * 1024
    downloaded_size = 0

    try:
        with ur.urlopen(url) as data:
            size = int(data.info()["Content-Length"])
            num_iter = int(size / chunk_size) + 2

            with open(path, "wb") as f:
                pbar = tqdm(range(num_iter))
                for _ in pbar:
                    chunk = data.read(chunk_size)
                    downloaded_size += len(chunk)
                    pbar.set_description(f"Downloaded {float(downloaded_size)/GBFACTOR:.2f} GB")
                    f.write(chunk)
    except Exception as exc:
        if osp.exists(path):
            os.remove(path)
        raise RuntimeError("Stopped downloading due to interruption.") from exc

    return path


def maybe_download_dataset(data_path):
    """Download dataset if it doesn't exist."""
    data_name = data_path.split("/")[-1]
    if data_name in ["FB15k", "FB15k-237", "NELL", "FB400k"]:
        stats_path = osp.join(data_path, "stats.txt")
        if not (osp.exists(data_path) and osp.exists(stats_path)):
            url = f"https://snap.stanford.edu/betae/{data_name}.zip"
            base_dir = osp.split(osp.abspath(data_path))[0]
            path = download_url(url, base_dir)
            extract_zip(path, base_dir)
            os.unlink(path)


def extract_zip(path, folder):
    r"""Extracts a zip archive to a specific folder.
    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
    """
    print("Extracting", path)
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(folder)


def thread_wrapped_func(func):
    """Wrapped func for torch.multiprocessing.Process.

    With this wrapper we can use OMP threads in subprocesses
    otherwise, OMP_NUM_THREADS=1 is mandatory.

    How to use:
    @thread_wrapped_func
    def func_to_wrap(args ...):
    """

    @wraps(func)
    def decorated_function(*args, **kwargs):
        result_queue = queue.Queue()

        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                exception = exc
                trace = traceback.format_exc()
            result_queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = result_queue.get()
        if exception is None:
            return result
        assert isinstance(exception, Exception)
        raise exception.__class__(trace)

    return decorated_function


def log_metrics(mode, step, metrics):
    """Print the evaluation logs."""
    for metric, value in metrics.items():
        logging.info("%s %s at step %d: %f", mode, metric, step, value)
