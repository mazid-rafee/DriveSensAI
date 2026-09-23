#!/usr/bin/env python3
"""Drowsiness window dataloader: session split, filtering, aug, standardization.

Window target rule (preserved): **last frame** of a causal window of length T.
``opening`` / ``closing`` endpoints are dropped; those frames may still appear
as temporal context inside kept windows.
"""

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from feature_contract import (  # noqa: E402
    CSV_SUFFIX,
    DROWSINESS_FEATURE_NAMES,
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
)
from label_contract import (  # noqa: E402
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    KEEP_CANONICAL_CLASSES,
)
from pre_process.dataloader import (  # noqa: E402
    ANNS_DIR,
    LANDMARKS_DIR,
    DMDGazeFrameDataset,
    create_dataloader,
)

DEFAULT_WINDOW_SIZE = 20
DEFAULT_SAMPLING_RATE_HZ = 15.0

BINARY_FEATURE_NAMES = ("face_detected", "left_eye_valid", "right_eye_valid")
POSE_FEATURE_NAMES = ("yaw", "pitch", "roll")
LEFT_RATIO_NAMES = ("left_eye_aspect_ratio", "left_eyelid_gap_ratio")
RIGHT_RATIO_NAMES = ("right_eye_aspect_ratio", "right_eyelid_gap_ratio")
LEFT_PUPIL_NAMES = ("left_pupil_rel_x", "left_pupil_rel_y")
RIGHT_PUPIL_NAMES = ("right_pupil_rel_x", "right_pupil_rel_y")
LEFT_EYE_CONTINUOUS = LEFT_RATIO_NAMES + LEFT_PUPIL_NAMES
RIGHT_EYE_CONTINUOUS = RIGHT_RATIO_NAMES + RIGHT_PUPIL_NAMES


def feature_index_map(
    feature_names: Sequence[str] = DROWSINESS_FEATURE_NAMES,
) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(feature_names)}


@dataclass
class AugmentationConfig:
    morphology_p: float = 0.5
    morphology_base_low: float = 0.85
    morphology_base_high: float = 1.15
    morphology_asym_low: float = 0.97
    morphology_asym_high: float = 1.03

    noise_p: float = 0.5
    noise_rho: float = 0.85
    pose_noise_std_radians: float = 0.015
    pupil_noise_std: float = 0.01
    ratio_noise_std_fraction: float = 0.02

    bias_p: float = 0.5
    yaw_bias_abs: float = 0.03
    pitch_bias_abs: float = 0.03
    roll_bias_abs: float = 0.02
    pupil_bias_abs: float = 0.03

    dropout_p: float = 0.15
    dropout_min_frames: int = 1
    dropout_max_frames: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DrowsinessFeatureAugmenter:
    """Train-time augmentations for a raw window shaped ``[T, 14]``."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str] = DROWSINESS_FEATURE_NAMES,
        config: Optional[AugmentationConfig] = None,
        train_feature_std: Optional[Mapping[str, float]] = None,
        seed: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        self.feature_names = list(feature_names)
        self.index = feature_index_map(self.feature_names)
        self.config = config or AugmentationConfig()
        self.train_feature_std = {
            name: float(train_feature_std.get(name, 1.0)) if train_feature_std else 1.0
            for name in self.feature_names
        }
        self.debug = bool(debug)
        self._rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def __call__(self, window: np.ndarray) -> np.ndarray:
        if window.ndim != 2 or window.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected window [T, {len(self.feature_names)}], got {window.shape}"
            )
        out = np.array(window, dtype=np.float64, copy=True)
        original = out.copy() if self.debug else None

        if self._rng.random() < self.config.morphology_p:
            out = self._morphology_scale(out)
        if self._rng.random() < self.config.noise_p:
            out = self._measurement_noise(out)
        if self._rng.random() < self.config.bias_p:
            out = self._calibration_bias(out)
        if self._rng.random() < self.config.dropout_p:
            out = self._landmark_dropout(out)

        out = enforce_feature_invariants(out, self.index)
        if self.debug and original is not None:
            self._print_debug(original, out)
        return out.astype(np.float32, copy=False)

    def _morphology_scale(self, window: np.ndarray) -> np.ndarray:
        cfg = self.config
        base = self._rng.uniform(cfg.morphology_base_low, cfg.morphology_base_high)
        left_scale = base * self._rng.uniform(
            cfg.morphology_asym_low, cfg.morphology_asym_high
        )
        right_scale = base * self._rng.uniform(
            cfg.morphology_asym_low, cfg.morphology_asym_high
        )
        for name in LEFT_RATIO_NAMES:
            window[:, self.index[name]] *= left_scale
        for name in RIGHT_RATIO_NAMES:
            window[:, self.index[name]] *= right_scale
        return window

    def _ar1_noise(self, length: int, std: float) -> np.ndarray:
        rho = float(self.config.noise_rho)
        eps = self._rng.normal(0.0, 1.0, size=length)
        noise = np.zeros(length, dtype=np.float64)
        if length == 0:
            return noise
        noise[0] = std * eps[0]
        scale = std * np.sqrt(max(0.0, 1.0 - rho * rho))
        for t in range(1, length):
            noise[t] = rho * noise[t - 1] + scale * eps[t]
        return noise

    def _measurement_noise(self, window: np.ndarray) -> np.ndarray:
        t_len = window.shape[0]
        idx = self.index
        face = window[:, idx["face_detected"]] > 0.5
        left_valid = (window[:, idx["left_eye_valid"]] > 0.5) & face
        right_valid = (window[:, idx["right_eye_valid"]] > 0.5) & face

        for name in POSE_FEATURE_NAMES:
            noise = self._ar1_noise(t_len, self.config.pose_noise_std_radians)
            window[:, idx[name]] += noise * face.astype(np.float64)

        for name in LEFT_PUPIL_NAMES:
            noise = self._ar1_noise(t_len, self.config.pupil_noise_std)
            window[:, idx[name]] += noise * left_valid.astype(np.float64)
        for name in RIGHT_PUPIL_NAMES:
            noise = self._ar1_noise(t_len, self.config.pupil_noise_std)
            window[:, idx[name]] += noise * right_valid.astype(np.float64)

        frac = self.config.ratio_noise_std_fraction
        for name in LEFT_RATIO_NAMES:
            std = frac * max(self.train_feature_std.get(name, 1.0), 1e-6)
            noise = self._ar1_noise(t_len, std)
            window[:, idx[name]] += noise * left_valid.astype(np.float64)
        for name in RIGHT_RATIO_NAMES:
            std = frac * max(self.train_feature_std.get(name, 1.0), 1e-6)
            noise = self._ar1_noise(t_len, std)
            window[:, idx[name]] += noise * right_valid.astype(np.float64)
        return window

    def _calibration_bias(self, window: np.ndarray) -> np.ndarray:
        cfg = self.config
        idx = self.index
        face = window[:, idx["face_detected"]] > 0.5
        left_valid = (window[:, idx["left_eye_valid"]] > 0.5) & face
        right_valid = (window[:, idx["right_eye_valid"]] > 0.5) & face

        yaw_b = self._rng.uniform(-cfg.yaw_bias_abs, cfg.yaw_bias_abs)
        pitch_b = self._rng.uniform(-cfg.pitch_bias_abs, cfg.pitch_bias_abs)
        roll_b = self._rng.uniform(-cfg.roll_bias_abs, cfg.roll_bias_abs)
        window[:, idx["yaw"]] += yaw_b * face.astype(np.float64)
        window[:, idx["pitch"]] += pitch_b * face.astype(np.float64)
        window[:, idx["roll"]] += roll_b * face.astype(np.float64)

        lx = self._rng.uniform(-cfg.pupil_bias_abs, cfg.pupil_bias_abs)
        ly = self._rng.uniform(-cfg.pupil_bias_abs, cfg.pupil_bias_abs)
        rx = self._rng.uniform(-cfg.pupil_bias_abs, cfg.pupil_bias_abs)
        ry = self._rng.uniform(-cfg.pupil_bias_abs, cfg.pupil_bias_abs)
        window[:, idx["left_pupil_rel_x"]] += lx * left_valid.astype(np.float64)
        window[:, idx["left_pupil_rel_y"]] += ly * left_valid.astype(np.float64)
        window[:, idx["right_pupil_rel_x"]] += rx * right_valid.astype(np.float64)
        window[:, idx["right_pupil_rel_y"]] += ry * right_valid.astype(np.float64)
        return window

    def _landmark_dropout(self, window: np.ndarray) -> np.ndarray:
        cfg = self.config
        t_len = window.shape[0]
        duration = int(
            self._rng.integers(cfg.dropout_min_frames, cfg.dropout_max_frames + 1)
        )
        duration = min(duration, t_len)
        start = int(self._rng.integers(0, t_len - duration + 1))
        end = start + duration
        which = self._rng.choice(["left", "right", "both"])
        idx = self.index
        if which in ("left", "both"):
            window[start:end, idx["left_eye_valid"]] = 0.0
            for name in LEFT_EYE_CONTINUOUS:
                window[start:end, idx[name]] = 0.0
        if which in ("right", "both"):
            window[start:end, idx["right_eye_valid"]] = 0.0
            for name in RIGHT_EYE_CONTINUOUS:
                window[start:end, idx[name]] = 0.0
        return window

    def _print_debug(self, original: np.ndarray, augmented: np.ndarray) -> None:
        print("--- aug debug: original mean ---")
        print(np.mean(original, axis=0))
        print("--- aug debug: augmented mean ---")
        print(np.mean(augmented, axis=0))


def enforce_feature_invariants(
    window: np.ndarray, index: Mapping[str, int]
) -> np.ndarray:
    """Clamp / zero features to satisfy schema invariants."""
    out = np.array(window, dtype=np.float64, copy=True)
    for name in BINARY_FEATURE_NAMES:
        col = index[name]
        out[:, col] = (out[:, col] >= 0.5).astype(np.float64)

    face = out[:, index["face_detected"]]
    no_face = face < 0.5
    if np.any(no_face):
        out[no_face, :] = 0.0

    for ratio_names, pupil_names, valid_name in (
        (LEFT_RATIO_NAMES, LEFT_PUPIL_NAMES, "left_eye_valid"),
        (RIGHT_RATIO_NAMES, RIGHT_PUPIL_NAMES, "right_eye_valid"),
    ):
        valid = out[:, index[valid_name]] >= 0.5
        invalid = ~valid
        for name in ratio_names:
            col = index[name]
            out[:, col] = np.maximum(out[:, col], 0.0)
            out[invalid, col] = 0.0
        for name in pupil_names:
            col = index[name]
            out[:, col] = np.clip(out[:, col], 0.0, 1.0)
            out[invalid, col] = 0.0

    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


@dataclass
class FeatureStandardizer:
    feature_names: List[str]
    mean: np.ndarray
    std: np.ndarray
    standardized_feature_mask: np.ndarray

    @classmethod
    def fit(
        cls,
        frames: np.ndarray,
        feature_names: Sequence[str] = DROWSINESS_FEATURE_NAMES,
    ) -> "FeatureStandardizer":
        """Fit on unaugmented training frames shaped ``[N, F]``."""
        if frames.ndim != 2 or frames.shape[1] != len(feature_names):
            raise ValueError(
                f"expected frames [N, {len(feature_names)}], got {frames.shape}"
            )
        index = feature_index_map(feature_names)
        mean = np.zeros(len(feature_names), dtype=np.float64)
        std = np.ones(len(feature_names), dtype=np.float64)
        mask = np.zeros(len(feature_names), dtype=np.bool_)

        binary = set(BINARY_FEATURE_NAMES)
        for name in feature_names:
            col = index[name]
            if name in binary:
                continue
            mask[col] = True
            if name in LEFT_EYE_CONTINUOUS:
                valid = frames[:, index["left_eye_valid"]] > 0.5
            elif name in RIGHT_EYE_CONTINUOUS:
                valid = frames[:, index["right_eye_valid"]] > 0.5
            else:
                valid = frames[:, index["face_detected"]] > 0.5
            values = frames[valid, col] if np.any(valid) else frames[:, col]
            mean[col] = float(np.mean(values)) if values.size else 0.0
            std_val = float(np.std(values)) if values.size else 1.0
            std[col] = std_val if std_val > 1e-6 else 1.0

        return cls(
            feature_names=list(feature_names),
            mean=mean,
            std=std,
            standardized_feature_mask=mask,
        )

    def transform(self, window: np.ndarray) -> np.ndarray:
        out = np.array(window, dtype=np.float64, copy=True)
        mask = self.standardized_feature_mask
        out[:, mask] = (out[:, mask] - self.mean[mask]) / self.std[mask]
        out = enforce_feature_invariants(out, feature_index_map(self.feature_names))
        return out.astype(np.float32, copy=False)

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        return {
            "feature_mean": self.mean.astype(np.float32),
            "feature_std": self.std.astype(np.float32),
            "standardized_feature_mask": self.standardized_feature_mask.astype(np.bool_),
        }

    @classmethod
    def from_checkpoint_dict(
        cls,
        payload: Mapping[str, Any],
        feature_names: Sequence[str] = DROWSINESS_FEATURE_NAMES,
    ) -> "FeatureStandardizer":
        return cls(
            feature_names=list(feature_names),
            mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            std=np.asarray(payload["feature_std"], dtype=np.float64),
            standardized_feature_mask=np.asarray(
                payload["standardized_feature_mask"], dtype=np.bool_
            ),
        )


class DrowsinessWindowDataset(Dataset):
    """Causal last-frame-labeled windows with optional train augmentation."""

    def __init__(
        self,
        frame_dataset: DMDGazeFrameDataset,
        *,
        window_size: int = DEFAULT_WINDOW_SIZE,
        augment: bool = False,
        augmenter: Optional[DrowsinessFeatureAugmenter] = None,
        standardizer: Optional[FeatureStandardizer] = None,
        require_contiguous_frame_ids: bool = True,
        verbose: bool = True,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if augment and augmenter is None:
            raise ValueError("augment=True requires an augmenter instance")

        self.frame_dataset = frame_dataset
        self.window_size = int(window_size)
        self.augment = bool(augment)
        self.augmenter = augmenter
        self.standardizer = standardizer
        self.require_contiguous_frame_ids = bool(require_contiguous_frame_ids)
        self.feature_names = list(frame_dataset.feature_names)
        self.class_to_idx = dict(CLASS_TO_IDX)
        self.idx_to_class = dict(IDX_TO_CLASS)
        self.session_keys = list(frame_dataset.session_keys)

        session_indices: Dict[str, List[int]] = defaultdict(list)
        for index, sample in enumerate(frame_dataset.samples):
            session_indices[sample["session_key"]].append(index)

        self._session_features: Dict[str, np.ndarray] = {}
        self._session_frame_ids: Dict[str, np.ndarray] = {}
        self._session_raw_labels: Dict[str, List[str]] = {}
        self._session_canonical: Dict[str, List[Optional[str]]] = {}

        before_counter: Counter = Counter()
        after_counter: Counter = Counter()
        removed_counter: Counter = Counter()
        per_session_before: Dict[str, Counter] = {}
        per_session_after: Dict[str, Counter] = {}
        per_session_removed: Dict[str, Counter] = {}

        self.samples: List[Dict[str, Any]] = []
        for session_key in self.session_keys:
            indices = session_indices.get(session_key, [])
            if not indices:
                continue
            feats = np.stack(
                [frame_dataset.samples[i]["features"].numpy() for i in indices],
                axis=0,
            ).astype(np.float32)
            frame_ids = np.asarray(
                [frame_dataset.samples[i]["frame_index"] for i in indices],
                dtype=np.int64,
            )
            raw_labels = [frame_dataset.samples[i]["raw_label_name"] for i in indices]
            canonical = [
                frame_dataset.samples[i]["canonical_label_name"] for i in indices
            ]
            self._session_features[session_key] = feats
            self._session_frame_ids[session_key] = frame_ids
            self._session_raw_labels[session_key] = raw_labels
            self._session_canonical[session_key] = canonical

            sess_before: Counter = Counter()
            sess_after: Counter = Counter()
            sess_removed: Counter = Counter()

            for local_index, canon in enumerate(canonical):
                # Candidate windows only (contiguous, in-session last-frame targets).
                if not self._window_is_valid(session_key, local_index):
                    continue
                raw = raw_labels[local_index]
                before_counter[raw] += 1
                sess_before[raw] += 1
                if canon is None:
                    removed_counter[raw] += 1
                    sess_removed[raw] += 1
                    continue
                if canon not in KEEP_CANONICAL_CLASSES:
                    raise RuntimeError(f"unexpected canonical {canon!r}")
                after_counter[canon] += 1
                sess_after[canon] += 1
                self.samples.append(
                    {
                        "session_key": session_key,
                        "local_index": int(local_index),
                        "frame_index": int(frame_ids[local_index]),
                        "label_index": int(self.class_to_idx[canon]),
                        "label_name": canon,
                        "raw_label_name": raw,
                    }
                )

            per_session_before[session_key] = sess_before
            per_session_after[session_key] = sess_after
            per_session_removed[session_key] = sess_removed

        self.counts_before_filter = dict(before_counter)
        self.counts_after_filter = {
            name: int(after_counter.get(name, 0)) for name in self.class_to_idx
        }
        self.removed_transition_windows = dict(removed_counter)
        self.per_session_counts_before = {
            k: dict(v) for k, v in per_session_before.items()
        }
        self.per_session_counts_after = {
            k: dict(v) for k, v in per_session_after.items()
        }
        self.per_session_removed = {k: dict(v) for k, v in per_session_removed.items()}
        self.class_counts = dict(self.counts_after_filter)

        if verbose:
            self._print_filter_summary()

    def _window_is_valid(self, session_key: str, local_index: int) -> bool:
        frame_ids = self._session_frame_ids[session_key]
        start = local_index - self.window_size + 1
        if start < 0:
            # Left-padded with first frame: only valid if session starts there.
            # Still one session; frame IDs for the real suffix must be contiguous.
            real_start = 0
        else:
            real_start = start
        ids = frame_ids[real_start : local_index + 1]
        if ids.size == 0:
            return False
        if self.require_contiguous_frame_ids and ids.size > 1:
            if not bool(np.all(np.diff(ids) == 1)):
                return False
        return True

    def _print_filter_summary(self) -> None:
        print("=== window target filter (last-frame label) ===")
        print(f"window_size: {self.window_size}")
        print(f"endpoint counts BEFORE filter (raw): {self.counts_before_filter}")
        print(
            f"opening/closing endpoints removed: {self.removed_transition_windows}"
        )
        print(
            f"endpoint counts AFTER filter (canonical): {self.counts_after_filter}"
        )
        print(f"kept windows: {len(self.samples)}")
        print("per-session after-filter canonical counts:")
        for key in self.session_keys:
            print(f"  {key}: {self.per_session_counts_after.get(key, {})}")

    def raw_training_frames(self) -> np.ndarray:
        """Unaugmented frame matrix for fitting standardization (all timeline rows)."""
        blocks = [
            self._session_features[key]
            for key in self.session_keys
            if key in self._session_features
        ]
        if not blocks:
            return np.zeros((0, len(self.feature_names)), dtype=np.float32)
        return np.concatenate(blocks, axis=0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        meta = self.samples[index]
        session_key = meta["session_key"]
        local_index = int(meta["local_index"])
        feats = self._session_features[session_key]
        start = local_index - self.window_size + 1
        if start >= 0:
            window = feats[start : local_index + 1].copy()
        else:
            available = feats[0 : local_index + 1]
            pad_n = self.window_size - int(available.shape[0])
            pad = np.repeat(available[:1], pad_n, axis=0)
            window = np.concatenate([pad, available], axis=0)

        if self.augment:
            assert self.augmenter is not None
            window = self.augmenter(window)
        if self.standardizer is not None:
            window = self.standardizer.transform(window)
        else:
            window = enforce_feature_invariants(
                window, feature_index_map(self.feature_names)
            ).astype(np.float32)

        if not np.isfinite(window).all():
            raise ValueError(
                f"non-finite window session={session_key} "
                f"frame_index={meta['frame_index']}"
            )
        label = torch.tensor(meta["label_index"], dtype=torch.long)
        return torch.from_numpy(np.asarray(window, dtype=np.float32)), label

    def get_metadata(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        return {
            "session_key": sample["session_key"],
            "frame_index": sample["frame_index"],
            "label_name": sample["label_name"],
            "label_index": sample["label_index"],
            "raw_label_name": sample["raw_label_name"],
            "window_size": self.window_size,
            "augment": self.augment,
        }


# Backward-compatible alias.
DMDGazeWindowDataset = DrowsinessWindowDataset


def split_indices_by_session(
    dataset: Dataset,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[str], List[str]]:
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


def build_train_val_window_datasets(
    frame_dataset: DMDGazeFrameDataset,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    val_ratio: float = 0.2,
    seed: int = 42,
    aug_config: Optional[AugmentationConfig] = None,
    verbose: bool = True,
) -> Tuple[DrowsinessWindowDataset, DrowsinessWindowDataset, FeatureStandardizer, Dict[str, Any]]:
    """Build filtered window datasets; fit standardizer on unaugmented train frames."""
    # Provisional split on frame-level samples to gather train session keys.
    train_idx, val_idx, train_sessions, val_sessions = split_indices_by_session(
        frame_dataset, val_ratio=val_ratio, seed=seed
    )
    del train_idx, val_idx

    train_sessions_set = set(train_sessions)
    # Restrict frame views by session for fitting stats / separate window builds.
    train_frame_indices = [
        i
        for i, s in enumerate(frame_dataset.samples)
        if s["session_key"] in train_sessions_set
    ]
    train_frames = np.stack(
        [frame_dataset.samples[i]["features"].numpy() for i in train_frame_indices],
        axis=0,
    ).astype(np.float32)
    standardizer = FeatureStandardizer.fit(train_frames, frame_dataset.feature_names)
    std_by_name = {
        name: float(standardizer.std[idx])
        for idx, name in enumerate(standardizer.feature_names)
    }
    aug_config = aug_config or AugmentationConfig()
    augmenter = DrowsinessFeatureAugmenter(
        feature_names=frame_dataset.feature_names,
        config=aug_config,
        train_feature_std=std_by_name,
        seed=seed,
    )

    full_windows = DrowsinessWindowDataset(
        frame_dataset,
        window_size=window_size,
        augment=False,
        standardizer=None,
        verbose=verbose,
    )
    # Rebuild train/val as separate datasets with correct augment flags.
    # Filter samples by session after constructing from full timeline.
    train_window = DrowsinessWindowDataset(
        frame_dataset,
        window_size=window_size,
        augment=True,
        augmenter=augmenter,
        standardizer=standardizer,
        verbose=False,
    )
    val_window = DrowsinessWindowDataset(
        frame_dataset,
        window_size=window_size,
        augment=False,
        augmenter=None,
        standardizer=standardizer,
        verbose=False,
    )
    train_window.samples = [
        s for s in train_window.samples if s["session_key"] in train_sessions_set
    ]
    val_window.samples = [
        s for s in val_window.samples if s["session_key"] in set(val_sessions)
    ]
    train_window.session_keys = list(train_sessions)
    val_window.session_keys = list(val_sessions)
    train_window.class_counts = dict(Counter(s["label_name"] for s in train_window.samples))
    val_window.class_counts = dict(Counter(s["label_name"] for s in val_window.samples))

    info = {
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "train_windows": len(train_window),
        "val_windows": len(val_window),
        "counts_before_filter": full_windows.counts_before_filter,
        "counts_after_filter": full_windows.counts_after_filter,
        "removed_transition_windows": full_windows.removed_transition_windows,
        "augmentation_config": aug_config.to_dict(),
        "val_ratio": val_ratio,
        "seed": seed,
        "window_size": window_size,
        "sampling_rate_hz": DEFAULT_SAMPLING_RATE_HZ,
    }
    if verbose:
        print("=== train/val window split ===")
        print(f"train sessions ({len(train_sessions)}): {train_sessions}")
        print(f"val sessions ({len(val_sessions)}): {val_sessions}")
        print(f"train windows: {len(train_window)} counts={train_window.class_counts}")
        print(f"val windows:   {len(val_window)} counts={val_window.class_counts}")
    return train_window, val_window, standardizer, info


def _worker_init_fn(seed: int):
    def _init(worker_id: int) -> None:
        worker_seed = int(seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        # Reseed dataset augmenter if present (train workers only).
        info = torch.utils.data.get_worker_info()
        if info is None:
            return
        dataset = info.dataset
        while isinstance(dataset, Subset):
            dataset = dataset.dataset
        augmenter = getattr(dataset, "augmenter", None)
        if augmenter is not None and hasattr(augmenter, "reseed"):
            augmenter.reseed(worker_seed)

    return _init


def make_train_val_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    *,
    batch_size: int = 256,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    device: Optional[object] = None,
    split_info: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Build shuffled train / unshuffled val loaders with seeded workers."""
    if pin_memory is None:
        if device is not None and getattr(device, "type", None) == "cuda":
            pin_memory = True
        else:
            pin_memory = bool(torch.cuda.is_available())

    g = torch.Generator()
    g.manual_seed(int(seed))
    worker_init = _worker_init_fn(seed) if num_workers > 0 else None
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": worker_init,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=g, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    info: Dict[str, Any] = dict(split_info or {})
    info.update(
        {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "pin_memory": bool(pin_memory),
            "device": str(device) if device is not None else None,
            "seed": int(seed),
        }
    )
    return train_loader, val_loader, info


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
    """Legacy helper for a pre-built window dataset (no per-split aug flags)."""
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
        "worker_init_fn": _worker_init_fn(seed) if num_workers > 0 else None,
    }
    g = torch.Generator()
    g.manual_seed(int(seed))
    train_loader = DataLoader(
        train_subset, shuffle=True, generator=g, **common
    )
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "ANNS_DIR",
    "CSV_SUFFIX",
    "DROWSINESS_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "LANDMARKS_DIR",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_SAMPLING_RATE_HZ",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "AugmentationConfig",
    "DrowsinessFeatureAugmenter",
    "FeatureStandardizer",
    "DMDGazeFrameDataset",
    "DrowsinessWindowDataset",
    "DMDGazeWindowDataset",
    "create_dataloader",
    "split_indices_by_session",
    "make_session_split_loaders",
    "make_train_val_loaders",
    "build_train_val_window_datasets",
    "seed_everything",
    "enforce_feature_invariants",
    "feature_index_map",
]
