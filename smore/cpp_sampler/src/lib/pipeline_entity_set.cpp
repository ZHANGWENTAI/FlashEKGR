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

#include <cstdint>
#include "pipeline_entity_set.h"

template<typename Dtype>
PipelineEntitySet<Dtype>::PipelineEntitySet()
{
    for (int i = 0; i < NUM_STAGES; ++i) {
        entity_sets[i].clear();
    }
}

template<typename Dtype>
PipelineEntitySet<Dtype>::~PipelineEntitySet()
{
    for (int i = 0; i < NUM_STAGES; ++i) {
        entity_sets[i].clear();
    }
}

template<typename Dtype>
bool PipelineEntitySet<Dtype>::has_any_entity(const Dtype* entity_ids, size_t num_entities)
{
    for (size_t i = 0; i < num_entities; ++i) {
        for (int stage = 0; stage < NUM_STAGES; ++stage) {
            if (entity_sets[stage].find(entity_ids[i]) != entity_sets[stage].end()) {
                return true;  // Found conflict
            }
        }
    }
    return false;  // No conflict
}

template<typename Dtype>
void PipelineEntitySet<Dtype>::insert_entities(int stage, const Dtype* entity_ids, size_t num_entities)
{
    if (stage < 0 || stage >= NUM_STAGES) {
        return;
    }
    for (size_t i = 0; i < num_entities; ++i) {
        entity_sets[stage].insert(entity_ids[i]);
    }
}

template<typename Dtype>
void PipelineEntitySet<Dtype>::clear_stage(int stage)
{
    if (stage >= 0 && stage < NUM_STAGES) {
        entity_sets[stage].clear();
    }
}

template<typename Dtype>
void PipelineEntitySet<Dtype>::move_stage(int from_stage, int to_stage)
{
    if (from_stage < 0 || from_stage >= NUM_STAGES || to_stage < 0 || to_stage >= NUM_STAGES) {
        return;
    }
    // Insert entities from from_stage to to_stage
    entity_sets[to_stage].insert(entity_sets[from_stage].begin(), entity_sets[from_stage].end());
    // Clear from_stage
    entity_sets[from_stage].clear();
}

template<typename Dtype>
int PipelineEntitySet<Dtype>::advance_stage(int current_stage)
{
    if (current_stage < 0 || current_stage >= NUM_STAGES) {
        return current_stage;
    }
    int next_stage = (current_stage + 1) % NUM_STAGES;
    move_stage(current_stage, next_stage);
    return next_stage;
}

template<typename Dtype>
size_t PipelineEntitySet<Dtype>::get_stage_size(int stage) const
{
    if (stage >= 0 && stage < NUM_STAGES) {
        return entity_sets[stage].size();
    }
    return 0;
}

// Explicit template instantiation
template class PipelineEntitySet<unsigned>;
template class PipelineEntitySet<uint64_t>;

