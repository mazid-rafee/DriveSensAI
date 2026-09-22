#!/usr/bin/env python3
"""DMD drowsiness dataloader facade with session-level train/val splitting.

Re-exports the frame-level dataset from ``pre_process.dataloader`` and adds
helpers that split by complete recording session (not by random frames) to
avoid leakage between adjacent video frames.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

# Allow ``python train.py`` from ``src/`` to resolve ``pre_process``.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pre_process.dataloader import (  # noqa: E402
    ANNS_DIR,
    DROWSINESS_FEATURE_NAMES,
    FEATURE_NAMES,
    LANDMARKS_DIR,
    DMDGazeFrameDataset,
    create_dataloader,
)

DEFAULT_WINDOW_SIZE = 20

__all__ = [
    "ANNS_DIR",
    "DROWSINESS_FEATURE_NAMES",
    "FEATURE_NAMES",
    "LANDMARKS_DIR",
    "DEFAULT_WINDOW_SIZE",
    "DMDGazeFrameDataset",
    "DMDGazeWindowDataset",
    "create_dataloader",
    "split_indices_by_session",
    "make_session_split_loaders",
]


class DMDGazeWindowDataset(Dataset):
    """Causal short-window view of ``DMDGazeFrameDataset``.

    Each item is ``(features[T, F], label)`` where the window ends on the
    labeled frame and is left-padded (repeat first available frame) when the
    session history is shorter than ``window_size``.
    """

    def __init__(
        self,
        frame_dataset: DMDGazeFrameDataset,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be positive, got {window_size}")

        self.frame_dataset = frame_dataset
        self.window_size = int(window_size)
        self.feature_names = list(frame_dataset.feature_names)
        self.class_to_idx = dict(frame_dataset.class_to_idx)
        self.idx_to_class = dict(frame_dataset.idx_to_class)
        self.session_keys = list(frame_dataset.session_keys)
        self.class_counts = dict(frame_dataset.class_counts)

        session_indices: Dict[str, List[int]] = defaultdict(list)
        for index, sample in enumerate(frame_dataset.samples):
            session_indices[sample["session_key"]].append(index)

        self._session_features: Dict[str, torch.Tensor] = {}
        self.samples: List[Dict[str, Any]] = []
        for session_key in self.session_keys:
            indices = session_indices.get(session_key, [])
            if not indices:
                continue
            features = torch.stack(
                [frame_dataset.samples[i]["features"] for i in indices],
                dim=0,
            )
            self._session_features[session_key] = features
            for local_index, global_index in enumerate(indices):
                sample = frame_dataset.samples[global_index]
                self.samples.append(
                    {
                        "session_key": session_key,
                        "local_index": int(local_index),
                        "frame_index": int(sample["frame_index"]),
                        "label_index": int(sample["label_index"]),
                        "label_name": sample["label_name"],
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        meta = self.samples[index]
        session_key = meta["session_key"]
        local_index = int(meta["local_index"])
        features = self._session_features[session_key]

        start = local_index - self.window_size + 1
        if start >= 0:
            window = features[start : local_index + 1]
        else:
            available = features[0 : local_index + 1]
            pad_n = self.window_size - int(available.shape[0])
            pad = available[:1].expand(pad_n, -1)
            window = torch.cat([pad, available], dim=0)

        if not bool(torch.isfinite(window).all()):
            raise ValueError(
                f"non-finite window at dataset index={index} "
                f"session={session_key} frame_index={meta['frame_index']}"
            )
        label = torch.tensor(meta["label_index"], dtype=torch.long)
        return window.clone(), label

    def get_metadata(self, index: int) -> Dict[str, Any]:
        """Return debugging metadata for sample ``index``."""
        sample = self.samples[index]
        return {
            "session_key": sample["session_key"],
            "frame_index": sample["frame_index"],
            "label_name": sample["label_name"],
            "label_index": sample["label_index"],
            "window_size": self.window_size,
        }


def split_indices_by_session(
    dataset: Dataset,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[str], List[str]]:
    """Split sample indices 80/20 by session key (never by shuffled frames).

    Returns
    -------
    train_indices, val_indices, train_sessions, val_sessions
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    sessions = list(dataset.session_keys)
    if len(sessions) < 2:
        raise ValueError(
            "Need at least 2 matched sessions for a train/val split, "
            f"found {len(sessions)}"
        )

    rng = random.Random(seed)
    shuffled = sessions[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    if n_val >= len(shuffled):
        n_val = len(shuffled) - 1
    val_sessions = sorted(shuffled[:n_val])
    train_sessions = sorted(shuffled[n_val:])
    val_set = set(val_sessions)
    train_set = set(train_sessions)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for index, sample in enumerate(dataset.samples):
        key = sample["session_key"]
        if key in train_set:
            train_indices.append(index)
        elif key in val_set:
            val_indices.append(index)
        else:
            raise RuntimeError(f"sample session {key!r} missing from split")

    if not train_indices or not val_indices:
        raise RuntimeError(
            "Session split produced an empty partition: "
            f"train={len(train_indices)} val={len(val_indices)}"
        )
    return train_indices, val_indices, train_sessions, val_sessions


def make_session_split_loaders(
    dataset: Dataset,
    *,
    batch_size: int = 256,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    device: Optional[object] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, object]]:
    """Build train/val DataLoaders from an 80/20 session split.

    Works with ``DMDGazeFrameDataset`` or ``DMDGazeWindowDataset``.

    When ``pin_memory`` is omitted, it defaults to ``True`` if CUDA is available
    (or if ``device`` is a CUDA ``torch.device``).
    """
    if pin_memory is None:
        if device is not None and getattr(device, "type", None) == "cuda":
            pin_memory = True
        else:
            pin_memory = bool(torch.cuda.is_available())

    train_indices, val_indices, train_sessions, val_sessions = (
        split_indices_by_session(dataset, val_ratio=val_ratio, seed=seed)
    )
    train_subset: Dataset = Subset(dataset, train_indices)
    val_subset: Dataset = Subset(dataset, val_indices)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_subset, shuffle=True, **common)
    val_loader = DataLoader(val_subset, shuffle=False, **common)
    info = {
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "val_ratio": val_ratio,
        "seed": seed,
        "pin_memory": bool(pin_memory),
        "device": str(device) if device is not None else None,
    }
    return train_loader, val_loader, info
