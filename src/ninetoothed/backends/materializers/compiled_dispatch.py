"""Guarded compiled dispatch for the installed Ascend Triton runtime."""
import os

class _CompiledDispatch:
    """Reuse Triton's own specialization cache without retaining tensor arguments.

    Restricted rollout for the three pointwise Ascend kernels. Unknown runtime
    contracts and JIT hooks use the ordinary JIT entry point.
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.explicit_stream = True

    def __getitem__(self, grid):
        kernel = self.kernel

        def run(*args, **kwargs):
            from triton.runtime import driver

            if (
                kernel.binder is None
                or kernel.pre_run_hooks
                or callable(grid)
                or not hasattr(kernel, "cache")
            ):
                return kernel[grid](*args, **kwargs)

            # Mirror the installed Triton binder and cache key exactly. This
            # includes pointer alignment, scalar specialization and options.
            kwargs["debug"] = kwargs.get("debug", False) or os.environ.get("TRITON_DEBUG", "0") == "1"
            bound, signature, constants, dynamic, excess = kernel.binder(*args, **kwargs)
            device = driver.active.get_current_device()
            key = "".join(signature) + str((constants, excess))
            compiled = kernel.cache[device].get(key)
            if compiled is None or hasattr(compiled, "result"):
                return kernel[grid](*args, **kwargs)

            missing = object()
            for (name, _), (value, namespace) in kernel.used_global_vals.items():
                if namespace.get(name, missing) != value:
                    # Ordinary JIT emits its normal diagnostic for mutation.
                    return kernel[grid](*args, **kwargs)

            padded = tuple(grid) + (1,) * (3 - len(grid))
            # The public CompiledKernel API resolves the current stream and
            # preserves launch metadata plus enter/exit hooks on every call.
            if self.explicit_stream:
                stream = driver.active.get_current_stream(device)
                compiled[padded](*dynamic, stream=stream)
            else:
                compiled[padded](*dynamic)
            return compiled

        return run


