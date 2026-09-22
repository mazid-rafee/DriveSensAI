#!/usr/bin/env python3
"""Train the DMD drowsiness causal TCN with session-level 80/20 validation.

Checkpoints are written under ``saved_weight/``:
  - ``best_accuracy.pt`` when validation accuracy improves
  - ``best_loss.pt`` when validation loss improves

Splits are performed by complete recording session (not by random frames) to
avoid leakage between adjacent video frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    make_session_split_loaders,
)
from device import (  # noqa: E402
    DEFAULT_GPU_ID,
    configure_cuda,
    resolve_device,
    to_device,
)
from loss import build_loss  # noqa: E402
from metrics import MulticlassMetricMeter  # noqa: E402
from model.model import GazeZoneTCN, build_model  # noqa: E402

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 42
DEFAULT_EARLY_STOPPING_PATIENCE = 7
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "saved_weight"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(
    model: GazeZoneTCN,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.train()
    meter = MulticlassMetricMeter(num_classes)
    progress = tqdm(
        loader,
        desc=f"train {epoch}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )
    for features, labels in progress:
        features = to_device(features, device, dtype=torch.float32)
        labels = to_device(labels, device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        meter.update(logits, labels, loss=loss)
        stats = meter.compute()
        progress.set_postfix(
            loss=f"{stats['loss']:.4f}",
            acc=f"{stats['accuracy']:.4f}",
        )
    return meter.compute()


@torch.no_grad()
def validate_one_epoch(
    model: GazeZoneTCN,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.eval()
    meter = MulticlassMetricMeter(num_classes)
    progress = tqdm(
        loader,
        desc=f"val   {epoch}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )
    for features, labels in progress:
        features = to_device(features, device, dtype=torch.float32)
        labels = to_device(labels, device, dtype=torch.long)
        logits = model(features)
        loss = criterion(logits, labels)
        meter.update(logits, labels, loss=loss)
        stats = meter.compute()
        progress.set_postfix(
            loss=f"{stats['loss']:.4f}",
            acc=f"{stats['accuracy']:.4f}",
        )
    return meter.compute()


def save_checkpoint(
    path: Path,
    *,
    model: GazeZoneTCN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    class_to_idx: Dict[str, int],
    split_info: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "arch": "tcn",
        "window_size": int(getattr(args, "window_size", DEFAULT_WINDOW_SIZE)),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "class_to_idx": class_to_idx,
        "feature_names": list(DROWSINESS_FEATURE_NAMES),
        "split_info": {
            "train_sessions": list(split_info["train_sessions"]),
            "val_sessions": list(split_info["val_sessions"]),
            "train_samples": int(split_info["train_samples"]),
            "val_samples": int(split_info["val_samples"]),
            "val_ratio": float(split_info["val_ratio"]),
            "seed": int(split_info["seed"]),
        },
        "args": vars(args),
    }
    torch.save(payload, path)


def plot_val_curves(
    history: Sequence[Dict[str, float]],
    output_path: Path,
    *,
    title: str = "Validation loss and accuracy",
) -> Path:
    """Draw validation loss and accuracy curves and save to ``output_path``."""
    if not history:
        raise ValueError("history is empty; nothing to plot")

    epochs = [int(row["epoch"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    val_acc = [float(row["val_accuracy"]) for row in history]

    best_loss_epoch = epochs[min(range(len(val_loss)), key=lambda i: val_loss[i])]
    best_acc_epoch = epochs[max(range(len(val_acc)), key=lambda i: val_acc[i])]
    best_loss = min(val_loss)
    best_acc = max(val_acc)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax_loss = plt.subplots(figsize=(8, 5))
    ax_acc = ax_loss.twinx()

    loss_line = ax_loss.plot(
        epochs, val_loss, color="#1f77b4", marker="o", linewidth=2, label="Val loss"
    )
    acc_line = ax_acc.plot(
        epochs,
        val_acc,
        color="#d62728",
        marker="s",
        linewidth=2,
        label="Val accuracy",
    )

    ax_loss.axvline(
        best_loss_epoch,
        color="#1f77b4",
        linestyle="--",
        alpha=0.5,
        label=f"Best loss @ {best_loss_epoch}",
    )
    ax_acc.axvline(
        best_acc_epoch,
        color="#d62728",
        linestyle=":",
        alpha=0.5,
        label=f"Best acc @ {best_acc_epoch}",
    )

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Validation loss", color="#1f77b4")
    ax_acc.set_ylabel("Validation accuracy", color="#d62728")
    ax_loss.tick_params(axis="y", labelcolor="#1f77b4")
    ax_acc.tick_params(axis="y", labelcolor="#d62728")
    ax_loss.set_title(title)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_xticks(epochs)

    lines = loss_line + acc_line
    labels = [line.get_label() for line in lines]
    ax_loss.legend(lines, labels, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    # print(
    #     f"wrote val curves: {output_path} "
    #     f"(best loss={best_loss:.4f} @ epoch {best_loss_epoch}, "
    #     f"best acc={best_acc:.4f} @ epoch {best_acc_epoch})"
    # )
    return output_path


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DMD drowsiness TCN (multiclass, session 80/20 split)."
    )
    parser.add_argument("--anns-dir", type=Path, default=ANNS_DIR)
    parser.add_argument("--landmarks-dir", type=Path, default=LANDMARKS_DIR)
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Causal input window length (default: 20).",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Stop after this many epochs without val-loss improvement.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory for best_accuracy.pt and best_loss.pt",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=DEFAULT_GPU_ID,
        help=f"CUDA device index (default: {DEFAULT_GPU_ID} -> cuda:{DEFAULT_GPU_ID})",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback when CUDA is unavailable (default: require CUDA).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Optional CrossEntropy label smoothing in [0, 1).",
    )
    parser.add_argument(
        "--curve-path",
        type=Path,
        default=None,
        help="Path for the validation loss/accuracy curve PNG "
        "(default: <checkpoint-dir>/val_loss_accuracy_curves.png).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.gpu, require_cuda=not args.allow_cpu)
    configure_cuda(device)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint_dir: {checkpoint_dir}")

    frame_dataset = DMDGazeFrameDataset(
        anns_dir=args.anns_dir,
        landmarks_dir=args.landmarks_dir,
        verbose=True,
    )
    if args.window_size < 1:
        raise ValueError(f"window_size must be positive, got {args.window_size}")
    dataset = DMDGazeWindowDataset(frame_dataset, window_size=args.window_size)
    print(f"TCN window_size: {args.window_size}")

    train_loader, val_loader, split_info = make_session_split_loaders(
        dataset,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        device=device,
    )
    print(f"DataLoader pin_memory={split_info['pin_memory']}")
    print(
        f"session split: train={split_info['train_samples']} "
        f"({len(split_info['train_sessions'])} sessions) | "
        f"val={split_info['val_samples']} "
        f"({len(split_info['val_sessions'])} sessions)"
    )
    print(f"train sessions: {split_info['train_sessions']}")
    print(f"val sessions:   {split_info['val_sessions']}")

    num_classes = len(frame_dataset.class_to_idx)
    model = build_model(
        num_classes=num_classes,
        input_dim=len(DROWSINESS_FEATURE_NAMES),
        dropout=args.dropout,
    ).to(device)
    print(f"model device: {next(model.parameters()).device}")
    print(f"trainable parameters: {model.num_trainable_parameters:,}")
    criterion = build_loss(label_smoothing=args.label_smoothing).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_val_accuracy = -float("inf")
    epochs_without_loss_improve = 0
    history = []

    best_loss_path = checkpoint_dir / "best_loss.pt"
    best_acc_path = checkpoint_dir / "best_accuracy.pt"
    curve_path = (
        args.curve_path.expanduser().resolve()
        if args.curve_path is not None
        else checkpoint_dir / "val_loss_accuracy_curves.png"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes,
            epoch,
            args.epochs,
        )
        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
            epoch,
            args.epochs,
        )
        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} "
            f"f1={row['train_macro_f1']:.4f} | "
            f"val loss={row['val_loss']:.4f} acc={row['val_accuracy']:.4f} "
            f"f1={row['val_macro_f1']:.4f} | "
            f"lr={row['lr']:.2e}"
        )

        # Refresh curves after every epoch so a crashed run still has a plot.
        plot_val_curves(history, curve_path)

        improved_loss = val_metrics["loss"] < best_val_loss
        improved_acc = val_metrics["accuracy"] > best_val_accuracy

        if improved_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_loss_improve = 0
            save_checkpoint(
                best_loss_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                class_to_idx=frame_dataset.class_to_idx,
                split_info=split_info,
                args=args,
            )
        else:
            epochs_without_loss_improve += 1

        if improved_acc:
            best_val_accuracy = val_metrics["accuracy"]
            save_checkpoint(
                best_acc_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                class_to_idx=frame_dataset.class_to_idx,
                split_info=split_info,
                args=args,
            )

        if epochs_without_loss_improve >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}: "
                f"no val-loss improvement for "
                f"{args.early_stopping_patience} epochs."
            )
            break

    history_path = checkpoint_dir / "train_history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"wrote history: {history_path}")
    plot_val_curves(history, curve_path)
    print(
        f"best val loss={best_val_loss:.6f} | "
        f"best val accuracy={best_val_accuracy:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
