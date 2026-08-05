#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model registry — register DNN model classes/factories and build them by name.

PyTorch models:     @register("name")     on a torch.nn.Module subclass
TensorFlow models:  @register_tf("name")  on a factory function (train_x, train_y, config) → model
"""

_REGISTRY = {}   # {name: {"factory": callable, "backend": "pytorch"|"tensorflow"}}


# ── Registration decorators ────────────────────────────────────────────────────

def register(name: str):
    """Decorator — registers a PyTorch nn.Module class under `name`."""
    def _decorator(cls):
        _REGISTRY[name] = {"factory": cls, "backend": "pytorch"}
        return cls
    return _decorator


def register_tf(name: str):
    """Decorator — registers a TensorFlow/Keras factory function under `name`.

    The factory must have the signature:
        factory(train_x, train_y, config) -> compiled keras Model
    """
    def _decorator(fn):
        _REGISTRY[name] = {"factory": fn, "backend": "tensorflow"}
        return fn
    return _decorator


# ── Query helpers ──────────────────────────────────────────────────────────────

def get_backend(name: str) -> str:
    """Return 'pytorch' or 'tensorflow' for a registered model name."""
    _check_registered(name)
    return _REGISTRY[name]["backend"]


def list_models():
    return list(_REGISTRY.keys())


def list_models_by_backend(backend: str):
    return [n for n, entry in _REGISTRY.items() if entry["backend"] == backend]


# ── Build helpers ──────────────────────────────────────────────────────────────

def build_model(name: str, input_size: int, hidden_size: int, output_size: int,
                num_layers: int, dropout: float, **kwargs):
    """Instantiate a registered PyTorch model by name using the standard interface."""
    _check_registered(name)
    entry = _REGISTRY[name]
    if entry["backend"] != "pytorch":
        raise ValueError(
            f"Model '{name}' is a TensorFlow model. "
            f"Call build_tf_model() instead, or use train_cell() which dispatches automatically."
        )
    return entry["factory"](
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout,
        **kwargs,
    )


def build_tf_model(name: str, train_x, train_y, config: dict):
    """Build a registered TensorFlow/Keras model by calling its factory with data arrays.

    The factory is called with (train_x, train_y, config) so it can infer
    input/output shapes from the actual training data.
    """
    _check_registered(name)
    entry = _REGISTRY[name]
    if entry["backend"] != "tensorflow":
        raise ValueError(
            f"Model '{name}' is a PyTorch model. "
            f"Call build_model() instead, or use train_cell() which dispatches automatically."
        )
    return entry["factory"](train_x, train_y, config)


# ── Internal ───────────────────────────────────────────────────────────────────

def _check_registered(name: str):
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Registered models: {list_models()}\n"
            f"Make sure to import the model module before calling the registry."
        )
