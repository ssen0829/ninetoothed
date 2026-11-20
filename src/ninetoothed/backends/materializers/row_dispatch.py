"""Validated specialization plans for bounded Ascend row kernels."""
import os


class RowLaunch:
    def __init__(self, handle, grid):
        import torch
        import torch_npu
        self.handle = handle
        if not handle._artifact.kernel_name.startswith('ntops_lab_row_block_'):
            raise ValueError('RowLaunch requires a bounded row-block artifact')
        if not handle._artifact.metadata.get('ssa_metadata', {}).get('schedule', {}).get('block_reductions', {}).get('enabled'):
            raise ValueError('RowLaunch requires the block-reduction schedule')
        self.grid = tuple(grid) + (1,) * (3 - len(grid))
        self.plans = {}
        abi = handle._compilation.launch_abi
        if len(abi.outputs) != 1:
            raise ValueError('RowLaunch requires one public output')
        self.output_index = abi.public_args.index(abi.outputs[0])
        specs = {spec.name: spec for spec in handle._compilation.kernel.tensors}
        self.tensor_indices = tuple(i for i, name in enumerate(abi.public_args)
                                    if specs[name].attrs.get("source_ndim", specs[name].ndim) > 0)
        self.scalar_indices = tuple(i for i in range(len(abi.public_args))
                                    if i not in self.tensor_indices)
        self.current_device = torch.npu.current_device
        self.device_context = torch.npu.device
        self.current_stream = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait',
                                      torch_npu._C._npu_getCurrentRawStream)

    def __call__(self, *args):
        kernel = self.handle._kernel
        if len(args) != len(self.tensor_indices) + len(self.scalar_indices):
            return self.handle(*args)
        tensors = tuple(args[index] for index in self.tensor_indices)
        if any(not hasattr(value, "data_ptr") for value in tensors):
            return self.handle(*args)
        x = tensors[0]
        out = args[self.output_index]
        device = self.current_device()
        if x.device.type == 'npu' and x.device.index != device:
            with self.device_context(x.device):
                return self(*args)
        try:
            debug = os.environ['TRITON_DEBUG'] == '1'
        except KeyError:
            debug = False
        # This key proves the same public tensor contract and Triton pointer
        # specialization; no tensor or pointer is retained in a launch plan.
        key = (device, debug, tuple(
            (type(t), t.shape, t.stride(), t.dtype, t.device,
             t.data_ptr() % 16)
            for t in tensors
        ), tuple((type(args[index]), args[index]) for index in self.scalar_indices))
        missing = object()
        globals_valid = all(namespace.get(name, missing) == value
                            for (name, _), (value, namespace)
                            in kernel.used_global_vals.items())
        plan = self.plans.get(key)
        if kernel.pre_run_hooks or not globals_valid:
            return self.handle(*args)
        if plan is None:
            from ninetoothed.compiler.runtime import _public_values, _bound_values

            compilation = self.handle._compilation
            public = _public_values(compilation.launch_abi, args, {},
                                    specs=compilation.kernel.tensors,
                                    device_types=("npu",))
            values, keepalive = _bound_values(compilation.launch_abi, public,
                                              scalar_mode="value")
            options = dict(BLOCK=1, num_warps=4, num_stages=3, debug=debug)
            kernel[self.grid](*values, **options)
            _, signature, constants, dynamic, excess = kernel.binder(*values, **options)
            cache_key = "".join(signature) + str((constants, excess))
            compiled = kernel.cache[device].get(cache_key)
            if compiled is None or hasattr(compiled, "result"):
                return out
            entries = []
            for value in dynamic:
                index = next((i for i, tensor in enumerate(tensors) if value is tensor), None)
                if index is not None:
                    entries.append((True, index))
                elif isinstance(value, (int, float, bool, str, type(None))):
                    entries.append((False, value))
                else:
                    return out
            if len(self.plans) >= 16:
                self.plans.pop(next(iter(self.plans)))
            self.plans[key] = (compiled[self.grid], tuple(entries))
            del keepalive
            return out
        runner, entries = plan
        dynamic = [tensors[value] if tensor else value for tensor, value in entries]
        stream = self.current_stream(device)
        # Public CompiledKernel runner retains launch enter/exit hooks and
        # profiler metadata, resolving the current stream on every invocation.
        runner(*dynamic, stream=stream)
        return out
