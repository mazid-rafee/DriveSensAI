#!/usr/bin/env python3
"""DMD drowsiness TCN inference helpers (importable) and optional CLI.

Importable API surface used by ``api.inference_service``:

* ``load_checkpoint`` — construct ``GazeZoneTCN``, load weights strictly
* ``predict_logits`` / ``predict_proba`` — run a ``[batch, T, F]`` tensor

The CLI entrypoint (``python inference.py``) only runs under ``__main__``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dataloader import (  # noqa: E402
    ANNS_DIR,
    DEFAULT_WINDOW_SIZE,
    DROWSINESS_FEATURE_NAMES,
    LANDMARKS_DIR,
    DMDGazeFrameDataset,
    DMDGazeWindowDataset,
)
from device import (  # noqa: E402
    DEFAULT_GPU_ID,
    configure_cuda,
    resolve_device,
    to_device,
)
from model.model import GazeZoneTCN, build_model  # noqa: E402

DEFAULT_CHECKPOINT = _SRC_DIR / "saved_weights" / "best_loss.pt"


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[GazeZoneTCN, Dict[str, Any], int]:
    """Load ``best_loss``-style checkpoint and return ``(model, ckpt, window)``.

    Uses ``load_state_dict(..., strict=True)``. Model is moved to ``device`` and
    set to ``eval()``.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint missing required key 'model_state_dict'")
    if "class_to_idx" not in checkpoint:
        raise KeyError("checkpoint missing required key 'class_to_idx'")

    class_to_idx: Dict[str, int] = dict(checkpoint["class_to_idx"])
    feature_names: List[str] = list(
        checkpoint.get("feature_names") or DROWSINESS_FEATURE_NAMES
    )
    if not feature_names:
        raise ValueError("checkpoint feature_names is empty")
    if len(class_to_idx) < 2:
        raise ValueError(
            f"class_to_idx must contain at least 2 classes, got {class_to_idx}"
        )

    args = checkpoint.get("args") or {}
    window_size = int(
        checkpoint.get("window_size")
        or args.get("window_size")
        or DEFAULT_WINDOW_SIZE
    )
    if window_size < 1:
        raise ValueError(f"invalid window_size={window_size}")

    model = build_model(
        num_classes=len(class_to_idx),
        input_dim=len(feature_names),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint, window_size


@torch.inference_mode()
def predict_logits(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return raw logits for ``features`` shaped ``[batch, T, F]``."""
    features = to_device(features, device, dtype=torch.float32)
    return model(features)


@torch.inference_mode()
def predict_proba(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return softmax class probabilities for ``features`` ``[batch, T, F]``."""
    logits = predict_logits(model, features, device)
    return torch.softmax(logits, dim=-1)


@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Score a DataLoader of ``(features, labels)`` batches (CLI / eval)."""
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    for features, labels in tqdm(loader, desc="infer", leave=False):
        features = to_device(features, device, dtype=torch.float32)
        labels = to_device(labels, device, dtype=torch.long)
        logits = model(features)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())
    return {
        "preds": torch.cat(all_preds, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "probs": torch.cat(all_probs, dim=0),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--anns-dir", type=Path, default=ANNS_DIR)
    parser.add_argument("--landmarks-dir", type=Path, default=LANDMARKS_DIR)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--gpu",
        type=int,
        default=DEFAULT_GPU_ID,
        help=f"CUDA device index (default: {DEFAULT_GPU_ID})",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback when CUDA is unavailable.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write prediction summary JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.gpu, require_cuda=not args.allow_cpu)
    configure_cuda(device)

    model, checkpoint, window_size = load_checkpoint(args.checkpoint, device)
    print(f"loaded checkpoint: {args.checkpoint}")
    print(f"checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"TCN window_size: {window_size}")
    print(f"model device: {next(model.parameters()).device}")

    frame_dataset = DMDGazeFrameDataset(
        anns_dir=args.anns_dir,
        landmarks_dir=args.landmarks_dir,
        class_to_idx=checkpoint["class_to_idx"],
        verbose=False,
    )
    dataset = DMDGazeWindowDataset(frame_dataset, window_size=window_size)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    outputs = run_inference(model, loader, device)
    preds = outputs["preds"]
    labels = outputs["labels"]
    accuracy = float((preds == labels).float().mean().item())
    print(f"samples: {len(dataset)}")
    print(f"accuracy: {accuracy:.6f}")
    print(f"preds device during compute: {device}")

    if args.output_json is not None:
        payload = {
            "checkpoint": str(args.checkpoint.resolve()),
            "device": str(device),
            "arch": "tcn",
            "window_size": window_size,
            "num_samples": int(len(dataset)),
            "accuracy": accuracy,
            "class_to_idx": checkpoint["class_to_idx"],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
