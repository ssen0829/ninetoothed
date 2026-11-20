import ast
import json
from types import SimpleNamespace

import pytest

from ninetoothed.ascendifier import Ascendifier
from ninetoothed.backends import Target, default_registry, emit
from ninetoothed.ir import ir_to_dict
from ninetoothed.targets import (
    resolve_target_context,
    runtime_device_types,
    validate_artifact_materialization,
)
from tests.test_backend_registry import _add_kernel


@pytest.mark.parametrize("platform", ("ascend-910b3", "ascend-910b4"))
def test_ascend_emits_python_with_platform_and_manifest(platform):
    context = resolve_target_context("ascend", platform=platform)
    assert runtime_device_types(SimpleNamespace(target=context)) == ("npu",)
    assert default_registry().get("ascend").capability.name == Target.ASCEND
    artifact = emit(_add_kernel(), "ascend", target_context=context)

    assert artifact.backend == Target.ASCEND
    assert artifact.language == "python/triton"
    assert artifact.primary_source_name.endswith(".ascend_triton.py")
    compile(artifact.primary_source, artifact.primary_source_name, "exec")
    tree = ast.parse(artifact.primary_source)
    assert artifact.entrypoint in {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "tl.store(" in artifact.primary_source
    assert artifact.metadata["target"]["platform"] == platform
    assert artifact.metadata["ssa_metadata"]["target_backend"] == "ascend"
    assert "ssa.ascend.optimize_schedule" in artifact.metadata["ssa_pass_trace"]
    manifest = json.loads(artifact.sources["add.ascend.json"])
    assert manifest["target"] == ir_to_dict(artifact.metadata["target"])
    validate_artifact_materialization(artifact, mode="jit")
    assert runtime_device_types(artifact) == ("npu",)
    validate_artifact_materialization(artifact, mode="aot")


@pytest.mark.parametrize("namespace", ("tl", "triton.language"))
@pytest.mark.parametrize("has_ascend", (True, False))
def test_ascendifier_transforms_generated_triton_syntax(namespace, has_ascend, monkeypatch):
    monkeypatch.setattr(
        "ninetoothed.ascendifier.importlib.util.find_spec",
        lambda name: object() if has_ascend else None,
    )
    source = f"""
from triton.language.extra import libdevice
x = {namespace}.load(ptr, other=None)
y = {namespace}.clamp(x, 0, 1).to({namespace}.float64)
"""
    result = Ascendifier().transform(source)
    compile(result, "ascend.py", "exec")
    module = "ascend" if has_ascend else "cann"
    assert f"from triton.language.extra.{module} import libdevice" in result
    assert "other=0.0" in result
    assert f"{namespace}.minimum({namespace}.maximum(x, 0), 1)" in result
    assert f"{namespace}.float32" in result


def test_ascendifier_preserves_autotune_configs():
    source = """
@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 512})], key=['x_size'])
def kernel(x):
    pass
"""
    result = Ascendifier().transform(source)
    assert "triton.Config({'BLOCK_SIZE': 512})" in result
    compile(result, "ascend.py", "exec")


@pytest.mark.parametrize('operation', ('fmod', 'copysign'))
def test_ascendifier_lowers_unsupported_external_math(operation):
    source = f'''
import triton
import triton.language as tl
from triton.language.extra import libdevice
@triton.jit
def kernel(x, y):
    return libdevice.{operation}(x, y)
'''
    result = Ascendifier().transform(source)
    compile(result, 'ascend_math.py', 'exec')
    assert f'return _nt_ascend_{operation}(x, y)' in result
    assert f'libdevice.{operation}(' not in result
    assert result.index(f'def _nt_ascend_{operation}') < result.index('def kernel')


def test_ascendifier_aligns_flat_masked_grid_only():
    source = '''
import triton
import triton.language as tl
@triton.jit
def kernel(X, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0)
    tl.store(X + offsets, -x, mask=mask)
def launch(X, N):
    grid = (triton.cdiv(N, 512),)
    kernel[grid](X, N, BLOCK=512)
'''
    result = Ascendifier().transform(source)
    compile(result, 'flat.py', 'exec')
    assert 'triton.cdiv(triton.cdiv(N, 512), _nt_vector_cores) * _nt_vector_cores' in result
    reduction = source.replace('x = tl.load', 'unused = tl.sum(tl.arange(0, BLOCK), 0)\n    x = tl.load')
    assert '_nt_vector_cores' not in Ascendifier().transform(reduction)
