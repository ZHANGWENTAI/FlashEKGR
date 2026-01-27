from contextlib import contextmanager
import unittest
from functools import wraps
import torch
from dataclasses import dataclass, is_dataclass, fields
from typing import List, Union, Dict, Tuple, Optional, Callable
import torch.nn as nn


_IS_GRAPH_CAPTURING = False

def _is_graph_capturing():
    """Query if currently capturing."""
    global _IS_GRAPH_CAPTURING
    return _IS_GRAPH_CAPTURING

def _set_capture_start():
    """Set graph capture has started."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = True

def _set_capture_end():
    """Set graph capture has ended."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = False

@contextmanager
def _swap_params(model_params_with_grad: List[torch.nn.Parameter], static_params: List[torch.Tensor]):
    original_datas = [p.data for p in model_params_with_grad]
    try:
        for p_model, p_static in zip(model_params_with_grad, static_params):
            p_model.data = p_static.data
        yield
    finally:
        for p_model, p_original in zip(model_params_with_grad, original_datas):
            p_model.data = p_original

@dataclass
class CudaGraphArgs:
    fwd_graph: torch.cuda.CUDAGraph
    static_inputs: List[torch.Tensor]
    static_params: List[torch.Tensor]
    static_outputs: Union[torch.Tensor, List[torch.Tensor]]
    bwd_graph: torch.cuda.CUDAGraph
    static_output_grads: Union[torch.Tensor, List[torch.Tensor]]
    static_param_grads: List[torch.Tensor]
    static_input_grads: List[torch.Tensor]


class _CudaGraphReplay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph_args: CudaGraphArgs, *runtime_args):
        # runtime_args is a flat tuple of all tensor inputs (data inputs + model params)
        for user_input, cudagraph_input in zip(runtime_args, graph_args.static_inputs + graph_args.static_params):
            cudagraph_input.copy_(user_input)

        graph_args.fwd_graph.replay()
        ctx.graph_args = graph_args

        return tuple(o.clone().detach() for o in graph_args.static_outputs)

    @staticmethod
    def backward(ctx, *grads):
        graph_args: CudaGraphArgs = ctx.graph_args

        for user_output_grad, cudagraph_output_grad in zip(grads, graph_args.static_output_grads):
            if user_output_grad is not None and cudagraph_output_grad is not None:
                cudagraph_output_grad.copy_(user_output_grad)

        graph_args.bwd_graph.replay()

        torch.cuda.synchronize() 

        param_grads = [w_grad.clone() if w_grad is not None else None for w_grad in graph_args.static_param_grads]
        input_grads = [d_grad.clone() if d_grad is not None else None for d_grad in graph_args.static_input_grads]
        
        # The number of returned gradients must match the number of inputs to forward().
        # forward inputs: (graph_args, *runtime_args)
        # backward outputs: (grad_for_graph_args, *grads_for_runtime_args)
        all_grads = tuple(input_grads) + tuple(param_grads)
        return (None,) + all_grads

class CudaGraphMgr:
    def __init__(self, num_warmup_steps=1):
        self.cuda_graphs_cache: Dict[str, CudaGraphArgs] = {}
        self.fwd_mem_pool: Optional[Tuple[int, int]] = None
        self.bwd_mem_pool: Optional[Tuple[int, int]] = None
        self.cache_count = 0
        self.stream = torch.cuda.Stream()
        self.num_warmup_steps = num_warmup_steps

    def run(self, func: Callable, *args, **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        key = self.__generate_key_for_cache(func, *args, **kwargs)
        if key not in self.cuda_graphs_cache:
            with torch.cuda.stream(self.stream):
                print("creating key:", key)
                _set_capture_start()
                self.__create_cuda_graph(func, *args, key=key, **kwargs)
                _set_capture_end()
            self.cache_count += 1
        graph_args = self.cuda_graphs_cache[key]
        
        runtime_inputs = self.__extract_tensors((args, kwargs))
        runtime_params = [p for p in args[0].parameters() if p.requires_grad]
        runtime_args = runtime_inputs + runtime_params
        
        return _CudaGraphReplay.apply(graph_args, *runtime_args)
    
    def __capture_fwd_graph(self, func: Callable, static_args: Tuple, static_kwargs: Dict) -> Tuple[torch.cuda.CUDAGraph, List[torch.Tensor]]:
        if self.fwd_mem_pool is None:
            self.fwd_mem_pool = torch.cuda.graph_pool_handle()
        fwd_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(fwd_graph, pool=self.fwd_mem_pool):
            static_outputs = func(*static_args, **static_kwargs)
        if isinstance(static_outputs, torch.Tensor):
            static_outputs = [static_outputs]
        return fwd_graph, list(static_outputs)

    def __capture_bwd_graph(self, model, static_outputs: List[torch.Tensor], static_inputs: List[torch.Tensor], static_params: List[torch.Tensor]) -> Tuple[torch.cuda.CUDAGraph, List, List, List]:
        if self.bwd_mem_pool is None:
            self.bwd_mem_pool = torch.cuda.graph_pool_handle()
        static_output_grads = [torch.zeros_like(o) for o in static_outputs if o.requires_grad]
        
        inputs_to_differentiate = [i for i in static_inputs if i.requires_grad] + \
                                  [p for p in model.parameters() if p.requires_grad]
        
        bwd_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(bwd_graph, pool=self.bwd_mem_pool):
            grads = torch.autograd.grad(
                outputs=[o for o in static_outputs if o.requires_grad],
                inputs=inputs_to_differentiate,
                grad_outputs=static_output_grads,
                only_inputs=True, allow_unused=True,
            )

        # 我们必须直接存储 `grads` 中的张量。
        num_inputs_with_grad = len([i for i in static_inputs if i.requires_grad])
        grads_for_inputs = grads[:num_inputs_with_grad]
        grads_for_params = grads[num_inputs_with_grad:]

        # 构建映射，以便将梯度放回正确的位置
        grad_map_inputs = {id(t): g for t, g in zip([i for i in static_inputs if i.requires_grad], grads_for_inputs)}
        static_input_grads = [grad_map_inputs.get(id(t)) for t in static_inputs]

        # 直接存储从 `grads` 中切片出来的参数梯度。
        # 这些张量就是 CUDA Graph 将在 replay 时更新的张量。
        static_param_grads = list(grads_for_params)

        return bwd_graph, static_output_grads, static_input_grads, static_param_grads

    def __create_cuda_graph(self, func: Callable, *args, key: str, **kwargs):
        model = args[0]
        model_params = [p for p in model.parameters() if p.requires_grad]
        static_params = [p.clone().detach().requires_grad_(True) for p in model_params]

        # Warmup. The wrapped func should be a forward-only function.
        for _ in range(self.num_warmup_steps):
            with _swap_params(model_params, static_params):
                outputs = func(*args, **kwargs)
                if isinstance(outputs, torch.Tensor): outputs = (outputs,)
                
                inputs_with_grad = [i for i in self.__extract_tensors((args, kwargs)) if i.requires_grad] + \
                                     [p for p in model.parameters() if p.requires_grad]
                
                # In forward-backward separated mode, we must do a dummy backward pass
                # to warm up the backward graph kernels.
                torch.autograd.grad(
                    outputs=tuple(o for o in outputs if o.requires_grad),
                    inputs=inputs_with_grad,
                    only_inputs=True, allow_unused=True,
                )

        # Capture
        runtime_inputs = self.__extract_tensors((args, kwargs))
        static_inputs = [t.clone() for t in runtime_inputs]
        for s_in, r_in in zip(static_inputs, runtime_inputs):
            s_in.requires_grad_(r_in.requires_grad)
        
        static_inputs_idx = [0]
        static_args = self.__replace_tensors(args, static_inputs, static_inputs_idx)
        static_kwargs = self.__replace_tensors(kwargs, static_inputs, static_inputs_idx)

        with _swap_params(model_params, static_params):
            fwd_graph, static_outputs = self.__capture_fwd_graph(func, static_args, static_kwargs)
            bwd_graph, static_output_grads, static_input_grads, static_param_grads = self.__capture_bwd_graph(
                model, static_outputs, static_inputs, static_params)
        
        self.cuda_graphs_cache[key] = CudaGraphArgs(
            fwd_graph=fwd_graph, static_inputs=static_inputs, static_params=static_params,
            static_outputs=static_outputs, bwd_graph=bwd_graph,
            static_output_grads=static_output_grads, static_param_grads=static_param_grads,
            static_input_grads=static_input_grads)

    def __extract_tensors(self, obj):
        tensors = []
        if isinstance(obj, torch.Tensor): tensors.append(obj)
        elif isinstance(obj, dict):
            for key in sorted(obj.keys()): tensors.extend(self.__extract_tensors(obj[key]))
        elif isinstance(obj, (list, tuple)):
            for item in obj: tensors.extend(self.__extract_tensors(item))
        elif is_dataclass(obj) and not isinstance(obj, type):
            for field in fields(obj): tensors.extend(self.__extract_tensors(getattr(obj, field.name)))
        return tensors
    
    def __extract_non_tensor_constants(self, obj):
        """
        Recursively extract all non-tensor constant values from the input object.
        """
        if isinstance(obj, torch.Tensor):
            return ()
        if isinstance(obj, (int, float, str, bool, type(None))):
            return (obj,)
        if isinstance(obj, dict):
            result = ()
            for key in sorted(obj.keys()):
                result += self.__extract_non_tensor_constants(obj[key])
            return result
        if isinstance(obj, (list, tuple)):
            result = ()
            for item in obj:
                result += self.__extract_non_tensor_constants(item)
            return result
        if is_dataclass(obj):
            result = ()
            for field in fields(obj):
                value = getattr(obj, field.name)
                result += self.__extract_non_tensor_constants(value)
            return result
        return ()

    def __replace_tensors(self, obj, static_inputs, static_input_idx):
        if isinstance(obj, torch.Tensor):
            result = static_inputs[static_input_idx[0]]
            static_input_idx[0] += 1
            return result
        elif isinstance(obj, dict): return {k: self.__replace_tensors(v, static_inputs, static_input_idx) for k, v in sorted(obj.items())}
        elif isinstance(obj, list): return [self.__replace_tensors(i, static_inputs, static_input_idx) for i in obj]
        elif isinstance(obj, tuple): return tuple(self.__replace_tensors(i, static_inputs, static_input_idx) for i in obj)
        elif is_dataclass(obj):
            return type(obj)(**{f.name: self.__replace_tensors(getattr(obj, f.name), static_inputs, static_input_idx) for f in fields(obj)})
        return obj

    def __generate_key_for_cache(self, func: Callable, *args, **kwargs):
        all_tensors = self.__extract_tensors((args, kwargs))
        tensor_shapes = [t.shape for t in all_tensors]
        shape_str = "_".join([",".join(map(str, shape)) for shape in tensor_shapes])
        non_tensor_constants = self.__extract_non_tensor_constants((args, kwargs))
        non_tensor_str = "_".join([str(x) for x in non_tensor_constants])
        return f"{func.__qualname__}_{shape_str}_{non_tensor_str}"


_CUDA_GRAPH_MGR = CudaGraphMgr()
def cuda_graph_mgr(): return _CUDA_GRAPH_MGR

def enable_cuda_graph():
    def decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if not torch.cuda.is_available() or _is_graph_capturing(): return func(*args, **kwargs)
            return cuda_graph_mgr().run(func, *args, **kwargs)
        return wrapped_func
    return decorator

class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)
    def forward(self, x): return self.fc2(self.relu(self.fc1(x)))

# The wrapped function must be forward-only
def forward_step(model, data, targets):
    output = model(data)
    return torch.nn.functional.mse_loss(output, targets)

@enable_cuda_graph()
def graphed_forward_step(model, data, targets): return forward_step(model, data, targets)

def disabled_graphed_step(model, data, targets): return forward_step(model, data, targets)

class TestCudaGraphMgr(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available(): self.skipTest("CUDA not available")
        global _CUDA_GRAPH_MGR
        _CUDA_GRAPH_MGR = CudaGraphMgr()
        torch.manual_seed(0)

    def test_graph_disabled_path(self):
        device, model = 'cuda', SimpleModel(16, 4).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial_params = [p.clone().detach() for p in model.parameters()]
        data, targets = torch.randn(8, 16, device=device), torch.randn(8, 4, device=device)
        optimizer.zero_grad()
        loss = disabled_graphed_step(model, data, targets)
        loss.backward()
        optimizer.step()
        self.assertEqual(cuda_graph_mgr().cache_count, 0)
        for p_i, p_f in zip(initial_params, model.parameters()): self.assertFalse(torch.equal(p_i, p_f.detach()))

    def test_training_loop_with_cuda_graph(self):
        device, model = 'cuda', SimpleModel(16, 4).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial_params = [p.clone().detach() for p in model.parameters()]

        num_steps, last_loss = 5, float('inf')
        data, targets = torch.randn(8, 16, device=device), torch.randn(8, 4, device=device)
        for i in range(num_steps):
            optimizer.zero_grad(set_to_none=True)
            
            # 1. Call the forward-only graphed function
            (loss,) = graphed_forward_step(model, data, targets)
            
            # 2. Call backward on the outside, which triggers _CudaGraphReplay.backward
            loss.backward()
            optimizer.step()

            self.assertEqual(cuda_graph_mgr().cache_count, 1)
            if i > 1:
                self.assertLess(loss.item(), last_loss)
            last_loss = loss.item()
            for param in model.parameters():
                self.assertIsNotNone(param.grad)

        for p_i, p_f in zip(initial_params, model.parameters()): self.assertFalse(torch.equal(p_i, p_f.detach()))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)