"""Adapter for Hugging Face OPT decoder FFN layers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM


def load_opt_causal_lm(
    model_id: str,
    revision: str = "main",
    cache_dir: str | None = None,
    device: str = "auto",
    hf_token: str | None = None,
) -> tuple[nn.Module, torch.device]:
    """Load an OPT causal language model from the Hugging Face Hub.

    OPT-125M is the default because its published configuration uses ReLU,
    making inactive FFN units exactly zero rather than threshold-dependent.
    """
    if device == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        token=hf_token,
    )
    model.to(resolved_device)
    model.eval()
    model.requires_grad_(False)

    if getattr(model.config, "model_type", None) != "opt":
        raise ValueError(
            f"This adapter supports OPT models, but {model_id!r} has model_type="
            f"{getattr(model.config, 'model_type', None)!r}. Add a model adapter before using it."
        )
    activation_name = getattr(model.config, "activation_function", None)
    if activation_name != "relu":
        raise ValueError(
            f"Expected ReLU FFN activations for exact-zero sparsity; found {activation_name!r}."
        )
    return model, resolved_device


def get_opt_ffn_layers(model: nn.Module) -> list[nn.Module]:
    """Return OPT decoder layers and validate the expected FFN module layout."""
    try:
        layers = list(model.model.decoder.layers)
    except AttributeError as exc:
        raise ValueError(
            "Could not locate OPT layers at model.model.decoder.layers. "
            "The Hugging Face model implementation may have changed."
        ) from exc
    if not layers or not all(hasattr(layer, "fc1") and hasattr(layer, "activation_fn") for layer in layers):
        raise ValueError("The loaded OPT model does not expose the expected fc1/activation_fn modules.")
    return layers


def layer_metadata(model: nn.Module, layers: list[nn.Module]) -> list[dict[str, Any]]:
    """Return human-readable FFN layer dimensions and activation information."""
    metadata: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        metadata.append(
            {
                "layer_index": index,
                "layer_name": f"layer_{index:02d}",
                "input_dim": int(layer.fc1.in_features),
                "ffn_dim": int(layer.fc1.out_features),
                "activation": getattr(model.config, "activation_function", "unknown"),
            }
        )
    return metadata
