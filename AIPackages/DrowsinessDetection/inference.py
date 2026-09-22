#!/usr/bin/env python3
"""Run DMD drowsiness TCN inference on CUDA (default ``cuda:1``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

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
from model.model import build_model  # noqa: E402


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_to_idx: Dict[str, int] = checkpoint["class_to_idx"]
    feature_names = checkpoint.get("feature_names", DROWSINESS_FEATURE_NAMES)
    input_dim = len(feature_names)
    args = checkpoint.get("args") or {}
    window_size = int(
        checkpoint.get("window_size")
        or args.get("window_size")
        or DEFAULT_WINDOW_SIZE
    )

    model = build_model(
        num_classes=len(class_to_idx),
        input_dim=input_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, window_size


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    all_preds = []
    all_labels = []
    all_probs = []
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


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "saved_weight"
        / "best_accuracy.pt",
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


def main(argv: Optional[list] = None) -> int:
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
