"""CPU test of the integrator time-point guard and bookkeeping (no model weights, no dataset).

`n_timesteps` is the number of TIME POINTS passed to VectorField.integrate; the integrator performs n_timesteps - 1 updates. A single time point used to
fail later with an unrelated KeyError; it is now rejected with a ValueError. The test calls the real `VectorField.integrate` on a stub `self` whose
`step` is a recorder, so it exercises the real time grid, the guard and the bookkeeping, not a copy of them.

usage: PYTHONPATH=<repo> python tests/unit/test_integrate_guard.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import dgl
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import omtra  # noqa: E402
import omtra.load.quick  # noqa: E402,F401  (importing the package entry point first avoids a circular import between omtra.data.graph and omtra.tasks)
from omtra.models.vector_field import VectorField  # noqa: E402

assert Path(omtra.__file__).resolve().is_relative_to(ROOT), 'omtra must come from this checkout'


class StubScheduler:
    def weights(self, t, task):
        return {'x': t}, {'x': 1 - t}

    def weight_derivative(self, t, task):
        return {'x': torch.ones_like(t)}, {'x': -torch.ones_like(t)}


def make_stub():
    calls = []

    def step(g, task, s_i, t_i, **kw):
        calls.append((float(t_i), float(s_i), bool(kw['last_step'])))
        return g, {}

    return SimpleNamespace(cat_temp_func=None, interpolant_scheduler=StubScheduler(), step=step), calls


def make_task():
    return SimpleNamespace(partial_modalities_fixed=[], node_modalities_present=[], edge_modalities_present=[])


def graph():
    g = dgl.heterograph({('a', 'e', 'a'): (torch.tensor([0]), torch.tensor([1]))}, num_nodes_dict={'a': 2})
    return dgl.batch([g])


# 1. fewer than two time points is rejected before anything else happens
for bad in (None, 0, 1, -3):
    stub, calls = make_stub()
    try:
        VectorField.integrate(stub, graph(), make_task(), {}, n_timesteps=bad)
    except ValueError as exc:
        assert 'time points' in str(exc), exc
        assert not calls, 'no step may run when the guard rejects the request'
    else:
        raise AssertionError(f'n_timesteps={bad!r} was accepted')

# 2. N time points = N-1 updates; the last update lands on t=1; bookkeeping agrees
for n in (2, 3, 26, 251):
    stub, calls = make_stub()
    VectorField.integrate(stub, graph(), make_task(), {}, n_timesteps=n)
    info = stub.last_integration_info
    assert len(calls) == n - 1, (n, len(calls))
    assert info['n_time_points'] == n and info['n_updates'] == n - 1 and info['n_network_evals'] == n - 1, info
    assert abs(calls[0][0]) < 1e-9 and abs(calls[-1][1] - 1.0) < 1e-6, calls[[0, -1]] if False else (calls[0], calls[-1])
    assert calls[-1][2] and not any(c[2] for c in calls[:-1]), 'only the final update is flagged last_step'
    assert abs(info['t_last_prediction'] - (1.0 - 1.0 / (n - 1))) < 1e-6, info

# 3. an explicit tspan bypasses the n_timesteps guard (callers who pass their own grid)
stub, calls = make_stub()
VectorField.integrate(stub, graph(), make_task(), {}, n_timesteps=1, tspan=torch.tensor([0.0, 0.5, 1.0]))
assert len(calls) == 2 and stub.last_integration_info['n_time_points'] == 3

print('test_integrate_guard: all checks passed')
