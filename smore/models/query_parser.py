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

"""
QueryParser: Automatically identify and merge parallelizable query branches.

This module provides functionality to analyze query structures and optimize
execution by parallelizing independent branches.
"""
from enum import Enum, auto
from typing import List, Tuple
import torch
from torch import nn

# Execution plan node types
class OpType(Enum):
    """Operation types for query execution plan."""
    LOAD_EMBED = auto()  # Retrieve entity embedding, inputs: entity ids, outputs: embedding
    RELATION_PROJ = auto()  # Relation projection, inputs: embedding, relation ids, outputs: embedding
    INTERSECTION = auto()  # Intersection operation, inputs: List[embeddings], outputs: embedding
    UNION = auto()  # Union operation, inputs: List[embeddings], outputs: embedding
    NEGATION = auto()  # Negation operation, inputs: embedding, outputs: embedding
    SPLIT = auto()  # Split operation, inputs: embedding, num_branches, outputs: List[embeddings]
    MERGE = auto()  # Merge operation, inputs: List[embeddings], outputs: embedding


class OpNode():
    def __init__(self, op_type: OpType, *args):
        super().__init__()
        self.op_type = op_type


class LoadEmbedNode(OpNode):
    def __init__(self, entity_idx: List[int]):
        super().__init__(OpType.LOAD_EMBED)
        self.entity_idx = entity_idx


class SplitNode(OpNode):
    def __init__(self, num_branches: int):
        super().__init__(OpType.SPLIT)
        self.num_branches = num_branches


class MergeNode(OpNode):
    def __init__(self):
        super().__init__(OpType.MERGE)


class RelationProjNode(OpNode):
    def __init__(self, cache_idx: int, relation_indices: List[int]):
        super().__init__(OpType.RELATION_PROJ)
        self.cache_idx = cache_idx
        self.relation_indices = relation_indices  # Index in queries matrix


class IntersectionNode(OpNode):
    def __init__(self):
        super().__init__(OpType.INTERSECTION)


class UnionNode(OpNode):
    def __init__(self):
        super().__init__(OpType.UNION)


class NegationNode(OpNode):
    def __init__(self, cache_idx: int):
        super().__init__(OpType.NEGATION)
        self.cache_idx = cache_idx


class QueryParser:
    def __init__(self):
        """Initialize the QueryParser."""
        self.cache_counter = 0

    def parse(self, query_structure: Tuple) -> List[OpNode]:
        """
        Parse query structure into execution plan.

        Args:
            query_structure: Query structure tuple

        Returns:
            List[OpNode]: Execution plan (list of operation nodes)
        """
        # Check if this is an all-relation path query
        if self._is_all_relation(query_structure):
            plan, _ = self._parse_path_query(query_structure, 0, 0)
            return plan

        # Check if this is a union query
        branch_plan, start_idx = self._parse_branch_query(query_structure[:-1], 0)
        tail_plan = self._parse_tail_query(query_structure[-1], start_idx)

        return branch_plan + tail_plan

    def _is_all_relation(self, query_structure: Tuple) -> bool:
        """Check if query structure is an all-relation path query."""
        if not isinstance(query_structure, tuple) or len(query_structure) != 2:
            return False
        if query_structure[0] != "e":
            return False
        if not isinstance(query_structure[-1], tuple):
            return False
        for ele in query_structure[-1]:
            if ele not in ["r", "n"]:
                return False
        return True

    def _parse_path_query(
        self, query_structure: Tuple, start_idx: int, branch_idx: int
    ) -> Tuple[List[OpNode], int]:
        """Parse a path query (all-relation query)."""
        idx = start_idx

        plan = []
        plan.append(LoadEmbedNode(entity_idx=[idx]))
        idx += 1
        for ele in query_structure[-1]:
            if ele == "r":
                plan.append(RelationProjNode(cache_idx=branch_idx, relation_indices=[idx]))
                idx += 1
            elif ele == "n":
                plan.append(NegationNode(cache_idx=branch_idx))
        return plan, idx
    
    def _parse_branch_query(self, query_structure: Tuple, start_idx: int) -> Tuple[List[OpNode], int]:
        """Parse a branch query."""
        branch_plans = []
        for branch_idx, branch in enumerate(query_structure):
            assert self._is_all_relation(branch), "Branch must be an all-relation query"
            branch_plan, start_idx = self._parse_path_query(branch, start_idx, branch_idx)
            branch_plans.append(branch_plan)
        
        branch_plan = self._find_common_prefix(branch_plans)

        return branch_plan, start_idx

    def _find_common_prefix(self, branch_plans: List[List[OpNode]]) -> List[OpNode]:
        """
        Find the common prefix of the branch plans.
        
        Args:
            branch_plans: List of branch plans, each is a list of OpNodes
            
        Returns:
            List[OpNode]: Common prefix operations that can be executed in parallel
        """
        # Find the minimum length to avoid index out of bounds
        min_length = min(len(plan) for plan in branch_plans)
        max_length = max(len(plan) for plan in branch_plans)
        
        common_prefix = []
        
        # Iterate through each position in parallel
        common_prefix_len = min_length  # Initialize to min_length (all positions match)
        for pos in range(min_length):
            # Get nodes at this position from all branches
            nodes_at_pos = [plan[pos] for plan in branch_plans]
            if all(node.op_type == OpType.LOAD_EMBED for node in nodes_at_pos):
                # Extract entity_idx from each node and flatten to 1D list
                entity_idxs = []
                for node in nodes_at_pos:
                    entity_idxs.extend(node.entity_idx)
                merged_node = LoadEmbedNode(entity_idx=entity_idxs)
                common_prefix.append(merged_node)
            elif all(node.op_type == OpType.RELATION_PROJ for node in nodes_at_pos):
                relation_indices = []
                for node in nodes_at_pos:
                    relation_indices.extend(node.relation_indices)
                merged_node = RelationProjNode(cache_idx=0, relation_indices=relation_indices)
                common_prefix.append(merged_node)
            elif all(node.op_type == OpType.NEGATION for node in nodes_at_pos):
                merged_node = NegationNode(cache_idx=0)
                common_prefix.append(merged_node)
            else:
                # Found first mismatch at position pos, common prefix length is pos
                common_prefix_len = pos
                break
        
        divergence_plan = []
        if common_prefix_len < max_length:
            common_prefix.append(SplitNode(num_branches=len(branch_plans)))
            for plan in branch_plans:
                divergence_plan.extend(plan[common_prefix_len:])
            common_prefix.append(MergeNode())

        return common_prefix + divergence_plan
    
    def _parse_tail_query(self, query_structure: Tuple, start_idx: int) -> List[OpNode]:
        """Parse a tail query."""
        assert all(ele in ["r", "n", "i", "u"] for ele in query_structure), "Tail must be a relation, negation, intersection, or union query"

        plan = []
        plan.append(LoadEmbedNode(entity_idx=[start_idx]))
        idx = start_idx + 1
        cache_idx = 0  # Cache index for tail operations
        for ele in query_structure[-1]:
            if ele == "r":
                plan.append(RelationProjNode(cache_idx=cache_idx, relation_indices=[idx]))
                idx += 1
            elif ele == "n":
                plan.append(NegationNode(cache_idx=cache_idx))
                # Negation doesn't increment idx, but uses the same cache_idx
        return plan
