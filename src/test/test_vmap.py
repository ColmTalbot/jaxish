import dataclasses
from collections import namedtuple
from functools import partial

import numpy as np
import pytest

from .. import vmap

# Test structures
Point = namedtuple("Point", ["x", "y"])


@dataclasses.dataclass
class State:
    position: float
    velocity: float


def test_vmap(xp):
    @vmap
    def f(x):
        return x * 2

    arr = xp.array([1, 2, 3])
    out = f(arr)
    assert np.allclose(out, arr * 2)


def test_vmap_tuple_output(xp):
    @vmap
    def f(x):
        return x, x + 1

    arr = xp.array([1, 2, 3])
    out1, out2 = f(arr)
    assert np.allclose(out1, arr)
    assert np.allclose(out2, arr + 1)


def test_vmap_multiple_args(xp):
    @vmap
    def f(x, y):
        return x + y

    arr1 = xp.array([1, 2, 3])
    arr2 = xp.array([4, 5, 6])
    out = f(arr1, arr2)
    assert np.allclose(out, arr1 + arr2)


def test_vmap_advanced_indexing(xp):
    @vmap
    def f(x):
        return x[0]

    arr = xp.array([[1, 2], [3, 4], [5, 6]])
    out = f(arr)
    assert np.allclose(out, arr[:, 0])


def test_vmap_specify_single_in_out_axis(xp):
    @partial(vmap, in_axes=1, out_axes=1)
    def f(x, y):
        return x * 2 + y

    arr = xp.array([[[1, 2], [3, 4], [5, 6]]])
    arr2 = xp.array([[1, 2, 3]])
    print(arr.shape, arr2.shape)
    out = f(arr, arr2)
    assert np.allclose(out, arr * 2 + arr2[:, :, None])


def test_vmap_specify_single_in_axis(xp):
    @partial(vmap, in_axes=1)
    def f(x, y):
        return x * 2 + y

    arr = xp.array([[[1, 2], [3, 4], [5, 6]]])
    arr2 = xp.array([[1, 2, 3]])
    print(arr.shape, arr2.shape)
    out = f(arr, arr2)
    assert np.allclose(out.swapaxes(0, 1), arr * 2 + arr2[:, :, None])


def test_vmap_specify_single_out_axis(xp):
    @partial(vmap, out_axes=1)
    def f(x, y):
        return x * 2 + y

    arr = xp.array([[1, 2], [3, 4], [5, 6]])
    arr2 = xp.array([1, 2, 3])
    print(arr.shape, arr2.shape)
    out = f(arr, arr2)
    print(out.shape)
    assert np.allclose(out.swapaxes(0, 1), arr * 2 + arr2[:, None])


def test_vmap_out_axis_out_of_range(xp):
    @partial(vmap, out_axes=5)
    def f(x, y):
        return x * 2 + y

    arr = xp.array([[1, 2], [3, 4], [5, 6]])
    arr2 = xp.array([1, 2, 3])
    with pytest.raises(ValueError):
        f(arr, arr2)


def test_not_args_number_mismatch_raises(xp):
    def f(x, y, z):
        return x * 2 + y + z

    args = xp.array([1, 2]), xp.array([3, 4]), xp.array([5, 6])

    with pytest.raises(ValueError):
        vmap(f, in_axes=(0, 1))(*args)
    with pytest.raises(ValueError):
        vmap(f, in_axes=(0, 1, 2, 3))(*args)


def test_vmap_mismatched_axes_raises(xp):
    @vmap
    def f(x, y):
        return x * 2 + y

    arr = xp.array([1, 2, 3])
    arr2 = xp.array([1, 3])

    with pytest.raises(ValueError):
        f(arr, arr2)


def test_vmap_specify_multiple_in_axis(xp):
    if xp.__name__ == "mlx.core":
        pytest.skip(reason="Not supported in mlx")

    @partial(vmap, in_axes=(1, 0))
    def f(x, y):
        return x * 2 + y

    arr = xp.array([[1, 2, 3], [4, 5, 6]])
    arr2 = xp.array([[1], [2], [3]])
    out = f(arr, arr2)
    assert np.allclose(out, arr.T * 2 + arr2)


def test_vmap_specify_multiple_out_axis(xp):
    if xp.__name__ == "mlx.core":
        pytest.skip(reason="Not supported in mlx")

    @partial(vmap, in_axes=(1, 1), out_axes=(1, 0))
    def f(x, y):
        return x * 2 + y, x * 3 + y

    arr = xp.array([[1, 2, 3], [4, 5, 6]])
    arr2 = xp.array([[1, 2, 3]])
    out, out2 = f(arr, arr2)
    assert np.allclose(out, arr * 2 + arr2)
    assert np.allclose(out2.swapaxes(0, 1), arr * 3 + arr2)


def test_vmap_namedtuple_input(xp):
    """Test vmap with namedtuple inputs."""
    if xp.__name__ == "mlx.core":
        pytest.skip(reason="Not supported in mlx")

    @vmap
    def f(point):
        print(point, type(point))
        return point.x + point.y

    points = Point(xp.arange(1, 4), xp.arange(2, 5))
    out = f(points)
    expected = xp.array([3, 5, 7])
    assert np.allclose(out, expected)


def test_vmap_namedtuple_output(xp):
    """Test vmap with namedtuple outputs."""

    @vmap
    def f(x, y):
        return Point(x * 2, y + 1)

    x = xp.array([1, 2, 3])
    y = xp.array([4, 5, 6])
    out = f(x, y)

    # Output should be a namedtuple with arrays
    assert hasattr(out, "x")
    assert hasattr(out, "y")
    assert np.allclose(out.x, xp.array([2, 4, 6]))
    assert np.allclose(out.y, xp.array([5, 6, 7]))


def test_vmap_dataclass_input(xp):
    """Test vmap with dataclass inputs."""
    if xp.__name__ == "mlx.core":
        pytest.skip(reason="Not supported in mlx")

    @vmap
    def f(state):
        return state.position + state.velocity

    states = [State(1.0, 0.5), State(2.0, 1.0), State(3.0, 1.5)]
    out = f(states)
    expected = xp.array([1.5, 3.0, 4.5])
    assert np.allclose(out, expected)


def test_vmap_dataclass_output(xp):
    """Test vmap with dataclass outputs."""
    if xp.__name__ == "mlx.core":
        pytest.skip(reason="Not supported in mlx")

    @vmap
    def f(pos, vel):
        return State(pos * 2, vel + 0.1)

    pos = xp.array([1.0, 2.0, 3.0])
    vel = xp.array([0.5, 1.0, 1.5])
    out = f(pos, vel)

    # Output should be a dataclass with arrays
    assert hasattr(out, "position")
    assert hasattr(out, "velocity")
    assert np.allclose(out.position, xp.array([2.0, 4.0, 6.0]))
    assert np.allclose(out.velocity, xp.array([0.6, 1.1, 1.6]))


def test_vmap_mixed_container_types(xp):
    """Test vmap with mixed container types in input/output."""

    @vmap
    def f(data):
        # data is a dict with mixed types
        point = data["point"]
        state = data["state"]
        return {
            "sum": point.x + point.y + state.position,
            "product": point.x * state.velocity,
        }

    data = [
        {"point": Point(1, 2), "state": State(10.0, 0.5)},
        {"point": Point(3, 4), "state": State(20.0, 1.0)},
        {"point": Point(5, 6), "state": State(30.0, 1.5)},
    ]

    out = f(data)

    assert "sum" in out
    assert "product" in out
    assert np.allclose(out["sum"], xp.array([13.0, 27.0, 41.0]))
    assert np.allclose(out["product"], xp.array([0.5, 3.0, 7.5]))


def test_vmap_nested_structures(xp):
    """Test vmap with deeply nested structures."""

    @vmap
    def f(nested):
        # nested = {'layer1': {'weights': Point, 'bias': float}, 'layer2': State}
        w = nested["layer1"]["weights"]
        b = nested["layer1"]["bias"]
        s = nested["layer2"]
        return w.x * s.position + w.y * s.velocity + b

    nested_data = {
        "layer1": {
            "weights": Point(xp.array([1, 2]), xp.array([3, 4])),
            "bias": xp.array([0.1, 1.1]),
        },
        "layer2": State(xp.array([10.0, 11.0]), xp.array([0.5, 1.5])),
    }

    out = f(nested_data)
    expected = xp.array([11.6, 29.1])
    assert np.allclose(out, expected)


def test_vmap_list_of_arrays(xp):
    """Test vmap with lists containing arrays."""

    @vmap
    def f(arr_list):
        return arr_list[0] + arr_list[1]

    data = xp.array([[1, 2], [3, 4], [5, 6]])

    out = f(data)
    expected = xp.array([3, 7, 11])
    assert np.allclose(out, expected)


def test_vmap_tuple_of_different_types(xp):
    """Test vmap with tuples containing different types."""
    if xp.__name__ == "mlx.core":
        pytest.skip(reason="Named tuple vmap not supported in mlx")

    @vmap
    def f(mixed_tuple):
        arr, point = mixed_tuple
        return arr.sum() + point.x + point.y

    data = (xp.array([1, 2]), Point(xp.array([5, 6]), xp.array([7, 8])))

    out = f(data)
    expected = xp.array([13, 16])
    assert np.allclose(out, expected)
