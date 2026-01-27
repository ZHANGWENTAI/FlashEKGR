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

from smore.common.util import eval_tuple
from smore.common.config import name_query_dict
from smore.models import model_list
from smore.models.query_parser import QueryParser

from .beta import BetaReasoning
from .box import BoxReasoning
from .complex import ComplexReasoning
from .distmult import DistmultReasoning
from .rotate import RotateReasoning
from .vec import VecReasoning


def build_model(args, nentity, nrelation, query_name_dict):
    tasks = args.tasks.split(".")
    has_neg = any("n" in task for task in tasks)

    # Validate union evaluation mode
    if args.evaluate_union == "DM":
        assert args.geo == "beta", "De Morgan's laws evaluation only supported for BetaE"
    
    if args.plan_mode == "lop":
        plan = {}
        for task in tasks:
            query_structures = name_query_dict[task]
            plan[task] = QueryParser().parse(query_structures)
    else: 
        plan = None

    # Common parameters for all models
    common_params = {
        "nentity": nentity,
        "nrelation": nrelation,
        "hidden_dim": args.hidden_dim,
        "gamma": args.gamma,
        "use_cuda": args.cuda,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "sparse_embeddings": args.sparse_embeddings,
        "sparse_device": args.sparse_device,
        "query_name_dict": query_name_dict,
        "optim_mode": args.optim_mode,
        "logit_impl": args.logit_impl,
        "plan": plan,
    }

    # Model class and mode configuration mapping
    model_config_map = {
        "box": (BoxReasoning, "box_mode", args.box_mode, "Q2B cannot handle queries with negation"),
        "rotate": (RotateReasoning, "rotate_mode", args.rotate_mode, "Rotate cannot handle queries with negation"),
        "complex": (ComplexReasoning, "complex_mode", args.complex_mode, "Complex cannot handle queries with negation"),
        "distmult": (
            DistmultReasoning,
            "distmult_mode",
            args.distmult_mode,
            "DistMult cannot handle queries with negation",
        ),
        "beta": (BetaReasoning, "beta_mode", args.beta_mode, None),
        "vec": (VecReasoning, "model_config", args.vec_mode, "GQE cannot handle queries with negation"),
    }

    # Get model configuration
    if args.geo in model_config_map:
        model_class, mode_key, mode_value, neg_error_msg = model_config_map[args.geo]

        # Check negation support
        if neg_error_msg and has_neg:
            raise AssertionError(neg_error_msg)

        # Build model-specific parameters
        model_params = common_params.copy()
        model_params[mode_key] = eval_tuple(mode_value)
        model = model_class(**model_params)
    else:
        # Fallback to model_list for custom models
        mod_class = getattr(model_list, args.geo)
        model_params = common_params.copy()
        model_params["model_config"] = eval_tuple(args.model_config)
        model = mod_class(**model_params)

    return model
