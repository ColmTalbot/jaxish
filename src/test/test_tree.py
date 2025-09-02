import dataclasses
from collections import namedtuple

import numpy as np
import pytest

from ..tree import tree_flatten, tree_unflatten

Point = namedtuple("Point", ["x", "y"])


@dataclasses.dataclass
class Person:
    name: str
    age: int
    scores: list = None


def test_tree_flatten_simple_list():
    tree = [1, 2, 3]
    leaves, treedef = tree_flatten(tree)
    assert leaves == [1, 2, 3]
    reconstructed = tree_unflatten(treedef, leaves)
    assert reconstructed == tree


def test_tree_flatten_simple_tuple():
    tree = (1, 2, 3)
    leaves, treedef = tree_flatten(tree)
    assert leaves == [1, 2, 3]
    reconstructed = tree_unflatten(treedef, leaves)
    assert reconstructed == tree


def test_tree_flatten_simple_dict():
    tree = {"a": 1, "b": 2}
    leaves, treedef = tree_flatten(tree)
    assert leaves == [1, 2]  # Sorted by key
    reconstructed = tree_unflatten(treedef, leaves)
    assert reconstructed == tree


def test_tree_flatten_nested():
    tree = {"a": [1, 2], "b": (3, {"c": 4})}
    leaves, treedef = tree_flatten(tree)
    assert leaves == [1, 2, 3, 4]
    reconstructed = tree_unflatten(treedef, leaves)
    assert reconstructed == tree


def test_tree_flatten_with_arrays():
    arr1 = np.array([1, 2])
    arr2 = np.array([3, 4])
    tree = [arr1, arr2]
    leaves, treedef = tree_flatten(tree)
    assert len(leaves) == 2
    np.testing.assert_array_equal(leaves[0], arr1)
    np.testing.assert_array_equal(leaves[1], arr2)
    reconstructed = tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(reconstructed[0], arr1)
    np.testing.assert_array_equal(reconstructed[1], arr2)


def test_tree_flatten_strings_as_leaves():
    tree = ["hello", "world"]
    leaves, treedef = tree_flatten(tree)
    assert leaves == ["hello", "world"]
    reconstructed = tree_unflatten(treedef, leaves)
    assert reconstructed == tree


def test_tree_flatten_empty_containers():
    tree = {"a": [], "b": (), "c": {}}
    leaves, treedef = tree_flatten(tree)
    assert leaves == []
    reconstructed = tree_unflatten(treedef, leaves)
    assert reconstructed == tree


def test_tree_flatten_single_leaf():
    tree = 42
    leaves, treedef = tree_flatten(tree)
    assert leaves == [42]
    reconstructed = tree_unflatten(treedef, leaves)
    assert reconstructed == tree


def test_tree_flatten_namedtuple():
    tree = Point(1, 2)
    leaves, treedef = tree_flatten(tree)
    assert leaves == [1, 2]
    reconstructed = tree_unflatten(treedef, leaves)
    assert reconstructed.x == tree.x
    assert reconstructed.y == tree.y
    assert type(reconstructed) is Point


def test_tree_flatten_dataclass():
    tree = Person("Alice", 30, [85, 92, 78])
    leaves, treedef = tree_flatten(tree)
    assert leaves == ["Alice", 30, 85, 92, 78]
    reconstructed = tree_unflatten(treedef, leaves)
    assert reconstructed.name == tree.name
    assert reconstructed.age == tree.age
    assert reconstructed.scores == tree.scores
    assert type(reconstructed) is Person


def test_tree_flatten_nested_with_dataclass():
    tree = {"person": Person("Bob", 25, [90, 95]), "location": Point(10, 20)}
    leaves, treedef = tree_flatten(tree)
    # Should have: location.x, location.y, person.age, person.name, person.scores[0], person.scores[1]
    assert len(leaves) == 6
    reconstructed = tree_unflatten(treedef, leaves)
    assert reconstructed["person"].name == "Bob"
    assert reconstructed["person"].age == 25
    assert reconstructed["person"].scores == [90, 95]
    assert reconstructed["location"].x == 10
    assert reconstructed["location"].y == 20


@pytest.mark.parametrize("backend", ["jax"])
def test_compare_with_jax(backend):
    """Compare with JAX's tree_flatten/tree_unflatten when available."""
    if backend == "jax":
        try:
            import jax.numpy as jnp
            from jax import tree_flatten as jax_tree_flatten
            from jax import tree_unflatten as jax_tree_unflatten
        except ImportError:
            pytest.skip("JAX not available")

        # Test cases that should match JAX behavior
        test_trees = [
            42,
            [1, 2, 3],
            (1, 2, 3),
            {"a": 1, "b": 2},
            {"a": [1, 2], "b": (3, 4)},
            [jnp.array([1, 2]), jnp.array([3, 4])],
        ]

        for tree in test_trees:
            # Compare flattening
            our_leaves, our_treedef = tree_flatten(tree)
            jax_leaves, jax_treedef = jax_tree_flatten(tree)

            # Leaves should be the same
            assert len(our_leaves) == len(jax_leaves)
            for our_leaf, jax_leaf in zip(our_leaves, jax_leaves):
                if isinstance(our_leaf, jnp.ndarray):
                    np.testing.assert_array_equal(our_leaf, jax_leaf)
                else:
                    assert our_leaf == jax_leaf

            # Test reconstruction
            our_reconstructed = tree_unflatten(our_treedef, our_leaves)
            jax_reconstructed = jax_tree_unflatten(jax_treedef, jax_leaves)

            # Reconstructed trees should match original
            if isinstance(tree, jnp.ndarray):
                np.testing.assert_array_equal(our_reconstructed, tree)
                np.testing.assert_array_equal(jax_reconstructed, tree)
            else:
                assert our_reconstructed == tree
                assert jax_reconstructed == tree


def test_tree_flatten_complex_nested():
    """Test a complex nested structure."""
    tree = {
        "layer1": {"weights": np.array([[1, 2], [3, 4]]), "bias": np.array([0.1, 0.2])},
        "layer2": {"weights": np.array([[5, 6]]), "bias": np.array([0.3])},
        "metadata": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "layers": ["dense", "dense"],
        },
    }

    leaves, treedef = tree_flatten(tree)
    reconstructed = tree_unflatten(treedef, leaves)

    # Check that all arrays are preserved
    np.testing.assert_array_equal(
        reconstructed["layer1"]["weights"], tree["layer1"]["weights"]
    )
    np.testing.assert_array_equal(
        reconstructed["layer1"]["bias"], tree["layer1"]["bias"]
    )
    np.testing.assert_array_equal(
        reconstructed["layer2"]["weights"], tree["layer2"]["weights"]
    )
    np.testing.assert_array_equal(
        reconstructed["layer2"]["bias"], tree["layer2"]["bias"]
    )

    # Check metadata
    assert reconstructed["metadata"]["learning_rate"] == 0.001
    assert reconstructed["metadata"]["batch_size"] == 32
    assert reconstructed["metadata"]["layers"] == ["dense", "dense"]
