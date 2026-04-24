"""tf.data.Dataset builders for text, image, and multimodal pipelines.

Assumes a TSV with columns ``CaptionText``, ``ImagePath``, ``MemeLabel``
(as produced by the original paper's preprocessing into ``top5_memes_tidy.tsv``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.data.preprocessing import tokenize_texts

TEXT_COL = "CaptionText"
LABEL_COL = "MemeLabel"
IMAGE_COL = "ImagePath"


def load_dataframe(
    tsv_path: str | Path,
    binary_classes: list[str] | None = None,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """Load tidy TSV, optionally restrict to two classes for binary mode."""
    df = pd.read_csv(tsv_path, sep="\t")

    for col in (TEXT_COL, IMAGE_COL, LABEL_COL):
        if col not in df.columns:
            raise ValueError(
                f"Expected column {col!r} in {tsv_path}; got {list(df.columns)}"
            )

    if binary_classes is not None:
        if len(binary_classes) != 2:
            raise ValueError("binary_classes must have exactly 2 entries")
        df = df[df[LABEL_COL].isin(binary_classes)].reset_index(drop=True)
        df[LABEL_COL] = (df[LABEL_COL] == binary_classes[0]).astype(np.int32)
        encoder = None
    else:
        encoder = LabelEncoder()
        df[LABEL_COL] = encoder.fit_transform(df[LABEL_COL].astype(str))

    return df, encoder


def split_dataframe(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    train_df, val_df = train_test_split(
        train_df, test_size=val_size / (1 - test_size), random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_text_dataset(
    df: pd.DataFrame,
    num_classes: int,
    pretrained: str,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    encoded = tokenize_texts(df[TEXT_COL].tolist(), pretrained, max_length)
    labels = df[LABEL_COL].to_numpy()
    if num_classes > 1:
        labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    else:
        labels = labels.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices(
        (
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
            labels,
        )
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=15)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_image_generator(
    df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    class_mode: str,
    shuffle: bool,
):
    datagen = ImageDataGenerator(
        preprocessing_function=vgg16.preprocess_input,
        rescale=1.0 / 255.0,
    )
    df = df.copy()
    df[LABEL_COL] = df[LABEL_COL].astype(str)
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=None,
        x_col=IMAGE_COL,
        y_col=LABEL_COL,
        class_mode=class_mode,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def build_multimodal_dataset(
    df: pd.DataFrame,
    num_classes: int,
    pretrained: str,
    max_length: int,
    image_size: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """Build a dataset yielding ((text_inputs, image), label) tuples."""
    encoded = tokenize_texts(df[TEXT_COL].tolist(), pretrained, max_length)

    paths = df[IMAGE_COL].to_numpy()

    def _load_image(path):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (image_size, image_size))
        img = vgg16.preprocess_input(img)
        return img / 255.0

    labels = df[LABEL_COL].to_numpy()
    if num_classes > 1:
        labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    else:
        labels = labels.astype(np.float32)

    input_ids_ds = tf.data.Dataset.from_tensor_slices(encoded["input_ids"])
    attn_ds = tf.data.Dataset.from_tensor_slices(encoded["attention_mask"])
    img_ds = tf.data.Dataset.from_tensor_slices(paths).map(
        _load_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    ds = tf.data.Dataset.zip(
        (
            {
                "input_ids": input_ids_ds,
                "attention_mask": attn_ds,
                "image": img_ds,
            },
            label_ds,
        )
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 2048), seed=15)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
