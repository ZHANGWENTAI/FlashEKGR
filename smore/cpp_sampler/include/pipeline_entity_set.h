// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PIPELINE_ENTITY_SET_H
#define PIPELINE_ENTITY_SET_H

#include <cstddef>
#include <unordered_set>
#include <vector>

template<typename Dtype>
class PipelineEntitySet {
public:
    PipelineEntitySet();
    ~PipelineEntitySet();

    // Batch lookup: check if any entity exists in any stage's hash table
    // Returns: true if any entity is found in any stage
    bool has_any_entity(const Dtype* entity_ids, size_t num_entities);

    // Batch insert: insert entities into specified stage's hash table
    // stage: 0=sampling, 1=loading, 2=computing
    void insert_entities(int stage, const Dtype* entity_ids, size_t num_entities);

    // Clear stage: remove all entities from specified stage's hash table
    void clear_stage(int stage);

    // Move stage: move entities from one stage to another
    // First inserts entities from from_stage to to_stage, then clears from_stage
    void move_stage(int from_stage, int to_stage);

    // Advance stage: move entities from current stage to next stage in cycle (0->1->2->0)
    // Returns the next stage index
    int advance_stage(int current_stage);

    // Get size of a stage
    size_t get_stage_size(int stage) const;

private:
    static const int NUM_STAGES = 3;
    std::unordered_set<Dtype> entity_sets[NUM_STAGES];
};

#endif

