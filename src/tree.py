import dataclasses
from collections.abc import Iterable

from array_api_compat import is_array_api_obj

__all__ = ["tree_flatten", "tree_unflatten"]


def _is_leaf(obj):
    """Check if an object is a leaf (array or non-iterable)."""
    return (
        is_array_api_obj(obj)
        or (not isinstance(obj, Iterable) and not _is_dataclass(obj))
        or isinstance(obj, (str, bytes))
    )


def _is_namedtuple(obj):
    """Check if an object is a namedtuple."""
    return (
        isinstance(obj, tuple)
        and hasattr(type(obj), "_fields")
        and hasattr(obj, "_fields")
    )


def _is_dataclass(obj):
    """Check if an object is a dataclass."""
    return dataclasses.is_dataclass(obj)


def tree_flatten(tree):
    """
    Flatten a pytree into a list of leaves and a tree definition.

    Args:
        tree: A pytree (nested structure of lists, tuples, dicts, and leaves)

    Returns:
        leaves: List of leaf values
        treedef: Tree definition for reconstruction
    """
    leaves = []

    def _flatten(obj, path=()):
        if _is_leaf(obj):
            leaves.append(obj)
            return ("leaf", len(leaves) - 1)
        elif _is_namedtuple(obj):
            children = []
            for i, child in enumerate(obj):
                children.append(_flatten(child, path + (i,)))
            return ("namedtuple", type(obj), children)
        elif _is_dataclass(obj):
            children = []
            fields = dataclasses.fields(obj)
            for field in fields:
                value = getattr(obj, field.name)
                children.append((field.name, _flatten(value, path + (field.name,))))
            return ("dataclass", type(obj), children)
        elif isinstance(obj, (list, tuple)):
            children = []
            for i, child in enumerate(obj):
                children.append(_flatten(child, path + (i,)))
            return (type(obj).__name__, children)
        elif isinstance(obj, dict):
            children = []
            keys = sorted(obj.keys())  # Sort for deterministic order
            for key in keys:
                children.append((key, _flatten(obj[key], path + (key,))))
            return ("dict", children)
        else:
            # Treat unknown iterables as leaves
            leaves.append(obj)
            return ("leaf", len(leaves) - 1)

    treedef = _flatten(tree)
    return leaves, treedef


def tree_unflatten(treedef, leaves):
    """
    Reconstruct a pytree from leaves and tree definition.

    Args:
        treedef: Tree definition from tree_flatten
        leaves: List of leaf values

    Returns:
        tree: Reconstructed pytree
    """
    leaf_iter = iter(leaves)

    def _unflatten(node):
        if node[0] == "leaf":
            return next(leaf_iter)
        elif node[0] == "namedtuple":
            namedtuple_type, children = node[1], node[2]
            values = [_unflatten(child) for child in children]
            return namedtuple_type(*values)
        elif node[0] == "dataclass":
            dataclass_type, children = node[1], node[2]
            field_values = {}
            for field_name, child in children:
                field_values[field_name] = _unflatten(child)
            return dataclass_type(**field_values)
        elif node[0] == "list":
            return [_unflatten(child) for child in node[1]]
        elif node[0] == "tuple":
            return tuple(_unflatten(child) for child in node[1])
        elif node[0] == "dict":
            return {key: _unflatten(child) for key, child in node[1]}
        else:
            raise ValueError(f"Unknown node type: {node[0]}")

    return _unflatten(treedef)
