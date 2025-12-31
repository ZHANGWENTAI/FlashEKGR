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

import os
import torch
from smore.cpp_sampler import libsampler
from smore.cpp_sampler import sampler_clib
from smore.common.util import name_query_dict

from tqdm import tqdm


def is_all_relation(query_structure):
    for ele in query_structure[-1]:
        if ele not in ["r", "n"]:
            return False
    return True


def has_negation(query_structure):
    for ele in query_structure[-1]:
        if ele == "n":
            return True
    return False


def build_query_tree(query_structure, fn_qt_create):
    if is_all_relation(query_structure):
        assert len(query_structure) == 2
        if query_structure[0] == "e":
            prev_node = fn_qt_create(libsampler.entity)
        else:
            prev_node = build_query_tree(query_structure[0], fn_qt_create)
        for i, c in enumerate(query_structure[-1]):
            if c == "r":
                cur_op = libsampler.relation
            else:
                assert c == "n"
                cur_op = libsampler.negation
            cur_root = fn_qt_create(libsampler.entity_set)
            cur_root.add_child(cur_op, prev_node)
            prev_node = cur_root
        return cur_root
    else:
        last_qt = query_structure[-1]
        node_type = libsampler.intersect
        if len(last_qt) == 1 and last_qt[0] == "u":
            node_type = libsampler.union
            query_structure = query_structure[:-1]
        sub_root = fn_qt_create(node_type)
        for c in query_structure:
            ch_node = build_query_tree(c, fn_qt_create)
            sub_root.add_child(libsampler.no_op, ch_node)
        return sub_root


class OnlineSampler(object):
    def __init__(
        self,
        kg,
        query_names,
        negative_sample_size,
        sample_mode,
        normalized_structure_prob,
        sampler_type="naive",
        share_negative=False,
        same_in_batch=False,
        weighted_answer_sampling=False,
        weighted_negative_sampling=False,
        nprefetch=10,
        num_threads=8,
    ):
        self.kg = kg
        kg_dtype = kg.dtype
        fn_qt_create = libsampler.create_qt32 if kg_dtype == "uint32" else libsampler.create_qt64
        query_structures = [name_query_dict[task] for task in query_names]
        self.query_structures = query_structures
        self.normalized_structure_prob = normalized_structure_prob
        assert len(normalized_structure_prob) == len(query_structures)
        self.negative_sample_size = negative_sample_size
        self.share_negative = share_negative
        self.same_in_batch = same_in_batch
        self.nprefetch = nprefetch
        if len(sample_mode) == 5:
            (
                self.rel_bandwidth,
                self.max_to_keep,
                self.weighted_style,
                self.structure_weighted_style,
                self.max_n_partial_answers,
            ) = sample_mode
            self.weighted_ans_sample = False
            self.weighted_neg_sample = False
        else:
            (
                self.rel_bandwidth,
                self.max_to_keep,
                self.weighted_style,
                self.structure_weighted_style,
                self.max_n_partial_answers,
                self.weighted_ans_sample,
                self.weighted_neg_sample,
            ) = sample_mode
        if self.rel_bandwidth <= 0:
            self.rel_bandwidth = kg.num_ent
        if self.max_to_keep <= 0:
            self.max_to_keep = kg.num_ent
        if self.max_n_partial_answers <= 0:
            self.max_n_partial_answers = kg.num_ent
        if self.structure_weighted_style == "wstruct":
            assert self.normalized_structure_prob is not None

        list_qt = []
        list_qt_nargs = []
        for qs in query_structures:
            if qs[0] == "<":  # inverse query
                assert is_all_relation(qs[1]) and not has_negation(qs[1])
                qt = build_query_tree(qs[1], fn_qt_create)
                qt.is_inverse = True
            else:
                qt = build_query_tree(qs, fn_qt_create)
            list_qt.append(qt)
            list_qt_nargs.append(qt.get_num_args())
        self.list_qt = list_qt
        self.list_qt_nargs = list_qt_nargs
        self.max_num_args = max(list_qt_nargs)
        no_search_list = []

        if sampler_type == "naive":
            sampler_cls = sampler_clib.naive_sampler(kg_dtype)
        elif sampler_type.startswith("sqrt"):
            sampler_cls = sampler_clib.rejection_sampler(kg_dtype)
            if "-" in sampler_type:
                no_search_list = [int(x) for x in sampler_type.split("-")[1].split(".")]
        elif sampler_type == "nosearch":
            sampler_cls = sampler_clib.no_search_sampler(kg_dtype)
        elif sampler_type == "edge":
            sampler_cls = sampler_clib.edge_sampler(kg_dtype)
            list_qt = query_names
        else:
            raise ValueError("Unknown sampler %s" % sampler_type)
        self.sampler_type = sampler_type
        self.sampler = sampler_cls(
            kg,
            list_qt,
            normalized_structure_prob,
            self.share_negative,
            self.same_in_batch,
            self.weighted_ans_sample,
            self.weighted_neg_sample,
            negative_sample_size,
            self.rel_bandwidth,
            self.max_to_keep,
            self.max_n_partial_answers,
            num_threads,
            no_search_list,
        )

        # Create PipelineEntitySet for tracking entities in pipeline
        if kg_dtype == "uint32":
            self.pipeline_entity_set = libsampler.create_pipeline_entity_set32()
        else:
            self.pipeline_entity_set = libsampler.create_pipeline_entity_set64()
        self.sampler.set_pipeline_entity_set(self.pipeline_entity_set)
        # Track current pipeline stage (0=sampling, 1=loading, 2=computing)
        self.current_pipeline_stage = 0

    def print_queries(self):
        self.sampler.print_queries()

    def sample_entities(self, weighted, num):
        entities = torch.LongTensor(num)
        self.sampler.sample_batch_entities(weighted, num, entities.numpy())
        return entities

    def set_seed(self, seed):
        self.sampler.set_seed(seed)

    def _extract_entity_ids_from_query_args(self, q_args, q_type):
        """
        Extract all entity IDs from query_args based on query structure.
        
        Args:
            q_args: Query arguments tensor [batch_size, num_args]
        q_type: Query type index
            
        Returns:
            numpy array of entity IDs or None
        """
        import numpy as np
        query_structure = self.query_structures[q_type]
        
        # Extract entity IDs based on query structure
        # For simple path queries like 1p, 2p: first arg is entity
        if is_all_relation(query_structure):
            entity_ids = q_args[:, 0].numpy()
            dtype = np.uint32 if self.kg.dtype == "uint32" else np.uint64
            return entity_ids.astype(dtype)
        
        # Complex query: extract entities from multiple positions
        num_args_per_query = self.list_qt_nargs[q_type]
        all_entity_ids = []
        
        for i in range(q_args.shape[0]):
            args = q_args[i, :num_args_per_query].numpy()
            entity_ids = []
            idx = 0
            
            # Parse query structure to extract entity positions
            if query_structure[-1][0] == "i":  # intersection
                num_branches = len(query_structure) - 1
                for branch_idx in range(num_branches):
                    branch = query_structure[branch_idx]
                    if isinstance(branch, tuple) and branch[0] == "e":
                        entity_ids.append(args[idx])
                        idx += 1
                        for ele in branch[-1]:
                            if ele == "r":
                                idx += 1
            elif query_structure[-1][0] == "u":  # union
                num_branches = len(query_structure) - 1
                for branch_idx in range(num_branches):
                    branch = query_structure[branch_idx]
                    if isinstance(branch, tuple) and branch[0] == "e":
                        entity_ids.append(args[idx])
                        idx += 1
                        for ele in branch[-1]:
                            if ele == "r":
                                idx += 1
            elif len(query_structure) == 3 and isinstance(query_structure[-1], tuple) and len(query_structure[-1]) > 0 and query_structure[-1][0] == "r":  # ip, 3ip, etc.
                num_branches = len(query_structure) - 1
                for branch_idx in range(num_branches):
                    branch = query_structure[branch_idx]
                    if isinstance(branch, tuple) and branch[0] == "e":
                        entity_ids.append(args[idx])
                        idx += 1
                        for ele in branch[-1]:
                            if ele == "r":
                                idx += 1
                # Tail query entity (if exists)
                tail_structure = query_structure[-1]
                if isinstance(tail_structure, tuple) and len(tail_structure) == 2 and tail_structure[0] == "e":
                    entity_ids.append(args[idx])
            
            all_entity_ids.extend(entity_ids)
        
        if len(all_entity_ids) > 0:
            dtype = np.uint32 if self.kg.dtype == "uint32" else np.uint64
            return np.array(all_entity_ids, dtype=dtype)
        return None

    def batch_generator(self, batch_size):
        self.sampler.prefetch(batch_size, self.nprefetch)
        uniform_weigths = torch.ones(batch_size)
        list_buffer = []
        for i in range(2):
            t_pos_ans = torch.LongTensor(batch_size)
            if self.share_negative:
                t_neg_ans = torch.LongTensor(1, self.negative_sample_size)
                t_is_neg_mat = torch.FloatTensor(batch_size, self.negative_sample_size)
            else:
                t_neg_ans = torch.LongTensor(batch_size, self.negative_sample_size)
                t_is_neg_mat = torch.FloatTensor(1, 2)
            t_weights = torch.FloatTensor(batch_size)
            t_arg_buffer = torch.LongTensor(batch_size, self.max_num_args)
            list_buffer.append((t_pos_ans, t_neg_ans, t_is_neg_mat, t_weights, t_arg_buffer))

        buf_idx = 0
        pos_ans, neg_ans, is_neg_mat, weights, arg_buffer = list_buffer[buf_idx]
        q_type = self.sampler.next_batch(
            pos_ans.numpy(), neg_ans.numpy(), weights.numpy(), is_neg_mat.numpy(), arg_buffer.numpy()
        )
        while True:
            next_buf_idx = 1 - buf_idx
            next_pos_ans, next_neg_ans, next_is_neg_mat, next_weights, next_arg_buffer = list_buffer[next_buf_idx]

            # T1 = time.perf_counter()
            next_q_type = self.sampler.next_batch(
                next_pos_ans.numpy(),
                next_neg_ans.numpy(),
                next_weights.numpy(),
                next_is_neg_mat.numpy(),
                next_arg_buffer.numpy(),
            )

            # T2 = time.perf_counter()
            # print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))

            if self.weighted_style == "u":
                weights = uniform_weigths
            pos_ans, neg_ans, is_neg_mat, weights, arg_buffer = list_buffer[buf_idx]
            q_args = arg_buffer[:, : self.list_qt_nargs[q_type]]
            q_structs = [self.query_structures[q_type]] * batch_size
            if self.sampler_type == "edge":
                is_neg_mat = None

            # Extract entity IDs and insert into current pipeline stage
            entity_ids = self._extract_entity_ids_from_query_args(q_args, q_type)
            if entity_ids is not None:
                self.pipeline_entity_set.insert_entities(self.current_pipeline_stage, entity_ids)

            yield pos_ans, neg_ans, is_neg_mat if self.share_negative else None, weights, q_args, q_structs
            q_type = next_q_type
            buf_idx = 1 - buf_idx


def has_negation(st):
    if isinstance(st, tuple):
        for c in st:
            if has_negation(c):
                return True
    else:
        assert isinstance(st, str)
        return st == "n"
    return False


def print_qt(qt, g, idx):

    node_type = str(qt.node_type).split(".")[-1]
    root_idx = str(idx)
    color = "#CCCCFF" if qt.sqrt_middle else "#FFFFFF"
    g.node(root_idx, node_type, fillcolor=color)
    idx += 1
    ch_list = []
    qt.get_children(ch_list)
    for ch in ch_list:
        ch_idx = idx
        idx = print_qt(ch, g, ch_idx)
        l = str(ch.parent_edge).split(".")[-1]
        if l == "no_op":
            l = ""
        s = "solid"
        if l == "negation":
            s = "dashed"
        g.edge(root_idx, str(ch_idx), label=l, style=s)
    return idx
