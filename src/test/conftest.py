import pytest

# All possible backends
ALL_BACKENDS = ["numpy", "jax", "mlx", "cupy"]
# Only differentiable backends
DIFFABLE_BACKENDS = ["jax", "mlx"]
NONDIFFABLE_BACKENDS = ["numpy", "cupy"]


def _get_array_backend(backend):
    match backend:
        case "numpy":
            from array_api_compat import numpy as xp
        case "jax":
            import jax.numpy as xp
        case "mlx":
            import mlx.core as xp
        case "cupy":
            import cupy as xp
        case _:
            raise ValueError(f"Unknown backend: {backend}")
    return xp


@pytest.fixture(params=ALL_BACKENDS)
def xp(request):
    pytest.importorskip(request.param)
    return _get_array_backend(request.param)


@pytest.fixture(params=DIFFABLE_BACKENDS)
def diffable(request):
    pytest.importorskip(request.param)
    return _get_array_backend(request.param)


@pytest.fixture(params=NONDIFFABLE_BACKENDS)
def undiffable(request):
    pytest.importorskip(request.param)
    return _get_array_backend(request.param)
