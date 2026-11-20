import weakref

import torch

from ninetoothed.compiler.runtime import _PreparedBoundValue


def test_contiguous_slice_keeps_pointer_and_storage_offset():
    storage = torch.arange(20)
    tensor = storage[3:11]
    bound = _PreparedBoundValue(tensor=weakref.ref(tensor), flatten_tensor=True)
    resolved, temporary = bound.resolve(flatten_tensor=True)
    assert resolved is tensor
    assert temporary is None
    assert resolved.storage_offset() == 3
    assert resolved.data_ptr() == tensor.data_ptr()


def test_strided_tensor_retains_storage_linearization():
    storage = torch.arange(20)
    tensor = storage[3:15:2]
    bound = _PreparedBoundValue(tensor=weakref.ref(tensor), flatten_tensor=True)
    resolved, temporary = bound.resolve(flatten_tensor=True)
    assert resolved is temporary
    assert resolved.stride() == (1,)
    torch.testing.assert_close(resolved, storage[3:])
