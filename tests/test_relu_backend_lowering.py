from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters.triton import TritonTarget


def test_relu_ascend_uses_cann_intrinsic():
    assert TritonTarget(backend=Target.ASCEND).call('relu', ('value',)) == 'libdevice.relu(value)'


def test_relu_triton_uses_portable_maximum():
    assert TritonTarget().call('relu', ('value',)) == 'tl.maximum(value, 0.0)'


def test_relu_cuda_and_tilelang_have_explicit_lowering():
    from ninetoothed.backends.emitters.cuda import CudaTarget
    from ninetoothed.backends.emitters.tilelang import TileLangTarget

    assert 'isnan(value)' in CudaTarget().call('relu', ('value',))
    assert TileLangTarget().call('relu', ('value',)) == 'T.max(value, 0.0)'
