"""Smoke tests: build every model + run a forward pass on dummy inputs.

Does NOT verify numerical correctness or training accuracy — only that the
factory dispatch, model wiring, and tensor shapes are consistent.

Note: first run downloads ``bert-base-uncased`` (~440 MB) from HuggingFace.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import tensorflow as tf

from src.config import load_config
from src.models.factory import build_model

CONFIG_FILES = [
    "configs/text/bert_binary.yaml",
    "configs/text/bert_multi.yaml",
    "configs/image/vgg_binary.yaml",
    "configs/image/vgg_multi.yaml",
    "configs/fusion/early_binary.yaml",
    "configs/fusion/early_multi.yaml",
    "configs/fusion/late_binary.yaml",
    "configs/fusion/late_multi.yaml",
]


def _dummy_inputs(model_cfg: dict, batch: int = 2):
    mtype = model_cfg["type"]
    max_length = int(model_cfg.get("max_length", 128))
    image_size = int(model_cfg.get("image_size", 224))

    text_inputs = [
        tf.ones([batch, max_length], dtype=tf.int32),
        tf.ones([batch, max_length], dtype=tf.int32),
    ]
    image_input = tf.random.uniform(
        [batch, image_size, image_size, 3], dtype=tf.float32
    )

    if mtype == "text":
        return text_inputs
    if mtype == "image":
        return image_input
    if mtype in {"early_fusion", "late_fusion"}:
        return text_inputs + [image_input]
    raise ValueError(mtype)


@pytest.mark.parametrize("cfg_path", CONFIG_FILES)
def test_build_and_forward(cfg_path: str):
    cfg = load_config(REPO_ROOT / cfg_path)
    model = build_model(cfg["model"])

    batch = 2
    x = _dummy_inputs(cfg["model"], batch=batch)
    y = model(x, training=False)

    expected_last_dim = int(cfg["model"]["num_classes"])
    assert y.shape[0] == batch, f"wrong batch dim for {cfg_path}: {y.shape}"
    assert y.shape[-1] == expected_last_dim, (
        f"wrong num_classes dim for {cfg_path}: {y.shape}, "
        f"expected last dim {expected_last_dim}"
    )
