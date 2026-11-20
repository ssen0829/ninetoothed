from dataclasses import replace

import pytest
import torch

import ninetoothed.language as ntl
from ninetoothed import Tensor, float32
from ninetoothed.backends.triton_reductions import TritonBlockReductions
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.compiler.passes import Context, default_spec
from ninetoothed.ir import ssa
from tests.utils import get_available_devices

PASS = "ssa.triton.block_reductions"


def _arrangement(x, out):
    x = x.tile((1, -1, -1))
    x.dtype = x.dtype.squeeze(0)
    out = out.tile((1, -1))
    out.dtype = out.dtype.squeeze(0)

    return x, out


def _application(x, out):
    weights = ntl.sum(x, axis=1)
    out = ntl.sum(x * weights[:, None], axis=0)  # noqa: F841


def _request(shape=(3, 8, 16), enabled=True):
    passes = default_spec("triton").passes

    return CompileRequest(
        arrangement=_arrangement,
        application=_application,
        tensors=(
            Tensor(shape=shape, dtype=float32),
            Tensor(shape=(shape[0], shape[2]), dtype=float32),
        ),
        backend="triton",
        pipeline=(*passes[:-1], PASS, passes[-1]) if enabled else None,
        num_warps=4,
        max_num_configs=1,
    )


def test_block_reduction_guards_preserve_the_original_schedule():
    compilation = DEFAULT_COMPILER.compile(_request(enabled=False))
    program = compilation.kernel.ssa
    context = Context(
        backend=compilation.target.backend,
        compiler_options={},
        kernel_metadata={},
        tensors=compilation.kernel.tensors,
    )
    transform = TritonBlockReductions()
    limited = replace(context, pass_options={PASS: {"max_block_elements": 32}})
    operations = program.blocks[0].operations
    unsupported = replace(
        program,
        blocks=(
            replace(
                program.blocks[0],
                operations=(
                    ssa.Operation(opcode="call.unknown"),
                    *operations,
                ),
            ),
        ),
    )

    for candidate, candidate_context in ((program, limited), (unsupported, context)):
        rejected = transform.run(candidate, candidate_context)
        schedule = dict(rejected.metadata["schedule"])
        assert not schedule.pop("block_reductions")["enabled"]
        assert schedule == dict(candidate.metadata.get("schedule", {}))
        assert replace(rejected, metadata=candidate.metadata) == candidate


@pytest.mark.parametrize("device", get_available_devices(backend="triton"))
def test_block_reduction_runtime(device):
    shape = (3, 8, 16)
    x = torch.randn((*shape[:-1], shape[-1] * 2), device=device)[..., ::2]
    output = torch.empty((shape[0], shape[2]), device=device)
    compilation = DEFAULT_COMPILER.compile(_request(shape))
    source = compilation.artifact.primary_source

    assert PASS not in default_spec("triton").passes
    assert PASS in compilation.pass_trace
    assert source.count("tl.sum(") == 2
    assert "for " not in source
    assert compilation.artifact.metadata["ssa_metadata"]["schedule"][
        "block_reductions"
    ]["enabled"]

    DEFAULT_COMPILER.materialize(compilation)(x, output)
    expected = (x.double() * x.double().sum(dim=2, keepdim=True)).sum(dim=1)
    torch.testing.assert_close(output, expected.float(), rtol=2e-5, atol=2e-5)


def _independent_arrangement(x, y, out):
    return x.tile((-1,)), y.tile((-1,)), out.tile((-1,))


def _independent_application(x, y, out):
    out = x + ntl.sum(y, axis=0)  # noqa: F841


@pytest.mark.parametrize("device", get_available_devices(backend="triton"))
def test_block_reduction_loads_each_input_domain(device):
    request = replace(
        _request(),
        arrangement=_independent_arrangement,
        application=_independent_application,
        tensors=(
            Tensor(shape=(8,), dtype=float32),
            Tensor(shape=(16,), dtype=float32),
            Tensor(shape=(8,), dtype=float32),
        ),
    )
    x = torch.zeros(8, device=device)
    y = torch.arange(16, device=device, dtype=torch.float32)
    out = torch.empty_like(x)

    for enabled in (False, True):
        compilation = DEFAULT_COMPILER.compile(
            request if enabled else replace(request, pipeline=None)
        )

        if enabled:
            assert compilation.artifact.metadata["ssa_metadata"]["schedule"][
                "block_reductions"
            ]["enabled"]

        DEFAULT_COMPILER.materialize(compilation)(x, y, out)
        torch.testing.assert_close(out, torch.full_like(out, 120), rtol=0, atol=0)


def _row_arrangement(x, out):
    return x.tile((16, -1)), out.tile((16, -1))


def _row_application(x, out):
    shifted = x - ntl.max(x, axis=1)[:, None]
    out = shifted - ntl.log(ntl.sum(ntl.exp(shifted), axis=1))[:, None]


@pytest.mark.parametrize("backend", ("triton", "ascend"))
def test_block_coordinates_preserve_two_dimensional_access(backend):
    passes = tuple(
        name for name in default_spec(backend).passes
        if name not in {
            "ssa.select_schedule", "ssa.triton.optimize_schedule",
            "ssa.ascend.optimize_schedule",
        }
    )
    request = CompileRequest(
        arrangement=_row_arrangement,
        application=_row_application,
        tensors=(Tensor(shape=(33, 32), dtype=float32, other=float("-inf")),
                 Tensor(shape=(33, 32), dtype=float32)),
        backend=backend,
        pipeline=(*passes[:-1], PASS, passes[-1]),
        max_num_configs=1,
    )
    compilation = DEFAULT_COMPILER.compile(request)
    source = compilation.artifact.primary_source
    assert compilation.artifact.metadata["ssa_metadata"]["schedule"][
        "block_reductions"
    ]["enabled"]
    assert "axis=1" in source
    # Recovering each lane from a flattened block defeats efficient 2D access.
    load_store = [line for line in source.splitlines()
                  if "tl.load(" in line or "tl.store(" in line]
    assert load_store
    assert all("// 32" not in line for line in load_store)


def _comparison_application(x, out):
    value = x - ntl.sum(x, axis=1)[:, None]
    value = ntl.where(value < 0.0, -value, value)
    out = ntl.where(value > 1.0, 1.0, value)


@pytest.mark.parametrize("backend", ("triton", "ascend"))
def test_block_reduction_comparison_selection(backend):
    passes = tuple(p for p in default_spec(backend).passes
                   if p not in {"ssa.select_schedule", "ssa.triton.optimize_schedule",
                                "ssa.ascend.optimize_schedule"})
    compilation = DEFAULT_COMPILER.compile(CompileRequest(
        arrangement=_row_arrangement, application=_comparison_application,
        tensors=(Tensor(shape=(33,32),dtype=float32),Tensor(shape=(33,32),dtype=float32)),
        backend=backend,pipeline=(*passes[:-1],PASS,passes[-1]),max_num_configs=1))
    assert compilation.artifact.metadata["ssa_metadata"]["schedule"]["block_reductions"]["enabled"]
    source = compilation.artifact.primary_source
    assert "tl.where(" in source
    assert "tl.sum(" in source
    assert "for " not in source


def _erf_application(x, out):
    value = x - ntl.sum(x, axis=1)[:, None]
    out = 0.5 * value * (1.0 + ntl.erf(value * 0.7071067811865476))


@pytest.mark.parametrize("backend", ("triton", "ascend"))
def test_block_reduction_erf(backend):
    passes = tuple(p for p in default_spec(backend).passes
                   if p not in {"ssa.select_schedule", "ssa.triton.optimize_schedule",
                                "ssa.ascend.optimize_schedule"})
    compilation = DEFAULT_COMPILER.compile(CompileRequest(
        arrangement=_row_arrangement, application=_erf_application,
        tensors=(Tensor(shape=(33,32),dtype=float32),Tensor(shape=(33,32),dtype=float32)),
        backend=backend,pipeline=(*passes[:-1],PASS,passes[-1]),max_num_configs=1))
    assert compilation.artifact.metadata["ssa_metadata"]["schedule"]["block_reductions"]["enabled"]
    assert "tl.erf(" in compilation.artifact.primary_source
