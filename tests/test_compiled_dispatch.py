import sys
from types import SimpleNamespace

from ninetoothed.backends.materializers.compiled_dispatch import _CompiledDispatch


def setup_dispatch(monkeypatch):
    calls = []
    state = {'device': 0, 'stream': 10, 'signature': 'aligned'}

    class Compiled:
        def __getitem__(self, grid):
            def run(*args, stream=None):
                calls.append(('compiled', grid, args, stream))
            return run

    compiled = Compiled()
    namespace = {'constant': 1}

    class Kernel:
        pre_run_hooks = []
        used_global_vals = {('constant', 0): (1, namespace)}
        cache = {0: {'aligned([], {})': compiled}, 1: {}}

        @staticmethod
        def binder(*args, **kwargs):
            return {}, [state['signature']], [], args, {}

        def __getitem__(self, grid):
            def run(*args, **kwargs):
                calls.append(('fallback', grid, args, kwargs))
            return run

    driver = SimpleNamespace(active=SimpleNamespace(
        get_current_device=lambda: state['device'],
        get_current_stream=lambda device: state['stream'],
    ))
    monkeypatch.setitem(sys.modules, 'triton.runtime', SimpleNamespace(driver=driver))
    proxy = _CompiledDispatch(Kernel())
    return proxy, state, calls, namespace


def test_dynamic_arguments_and_current_stream(monkeypatch):
    proxy, state, calls, _ = setup_dispatch(monkeypatch)
    proxy.explicit_stream = True
    first, second = object(), object()
    proxy[(4,)](first)
    state['stream'] = 20
    proxy[(4,)](second)
    assert calls == [('compiled', (4, 1, 1), (first,), 10),
                     ('compiled', (4, 1, 1), (second,), 20)]


def test_changed_specialization_and_device_fall_back(monkeypatch):
    proxy, state, calls, _ = setup_dispatch(monkeypatch)
    state['signature'] = 'unaligned'
    proxy[(4,)](object())
    state['signature'] = 'aligned'; state['device'] = 1
    proxy[(4,)](object())
    assert [c[0] for c in calls] == ['fallback', 'fallback']


def test_global_mutation_and_pre_run_hooks_fall_back(monkeypatch):
    proxy, state, calls, namespace = setup_dispatch(monkeypatch)
    namespace['constant'] = 2
    proxy[(4,)](object())
    namespace['constant'] = 1
    proxy.kernel.pre_run_hooks = [lambda: None]
    proxy[(4,)](object())
    assert [c[0] for c in calls] == ['fallback', 'fallback']
