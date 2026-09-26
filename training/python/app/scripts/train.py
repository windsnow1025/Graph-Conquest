"""Train the model on binary sample data exported by TypeScript (v11, 11 heads).

Objectives:
  imitation  every head fits its label (weights non-negative; phase 1)
  ppo        clipped policy surrogate on the sampled heads with signed,
             standardized advantages and the behavior log-probabilities taken
             from the starting weights, value MSE, weighted BCE on the
             threshold-decoded heads (phases 2 and 3)

Usage:
  uv run python -m app.scripts.train --objective imitation --data samples.bin --model path/to/model/ --fresh
  uv run python -m app.scripts.train --objective ppo --data phase2.bin --model training/model/phase2 --epochs 4 --lr 0.0001
"""
import argparse
import time

import numpy as np
import torch

from app.config import NUM_ACTION_TYPES, OFF_ACTION_TYPE, OFF_POLICY_WEIGHT
from app.model import GraphConquestNN
from app.data_io import read_samples, read_multiple
from app.trainer import train_epoch, eval_loss, prepare_ppo, head_names
from app.export_tfjs import export_model, import_tfjs_weights


def _fmt(total, heads, names):
    if len(names) <= 6:
        return f"loss={total:.4f} " + " ".join(f"{name}={v:.6f}" for name, v in zip(names, heads))
    # Line 1: loss + non-battle heads (val act mfr dfr rec mov)
    line1 = " ".join(f"{name}={v:.6f}" for name, v in zip(names[:6], heads[:6]))
    # Line 2: battle heads (btgt bsel cfr kfr ret)
    line2 = " ".join(f"{name}={v:.6f}" for name, v in zip(names[6:], heads[6:]))
    return f"loss={total:.4f} {line1}\nbattle: {line2}"


def main():
    parser = argparse.ArgumentParser(description="Train Graph Conquest NN v11")
    parser.add_argument("--objective", choices=["imitation", "ppo"], required=True)
    parser.add_argument("--clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--data", nargs="+", required=True, help="Binary sample files")
    parser.add_argument("--model", required=True, help="TF.js model directory (read + write)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fresh", action="store_true", help="Train from scratch (ignore existing weights)")
    parser.add_argument("--balance-actions", type=float, default=0.0,
                        help="Weight the action-type loss by inverse class frequency to this power (0 = off; imitation: the rare MOVE label)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    t0 = time.time()
    if len(args.data) == 1:
        data = read_samples(args.data[0])
    else:
        data = read_multiple(args.data)
    print(f"Loaded {data.shape[0]} samples in {time.time() - t0:.1f}s")

    model = GraphConquestNN().to(device)
    if not args.fresh:
        import_tfjs_weights(model, args.model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.objective == "ppo":
        if args.fresh:
            raise SystemExit("ppo needs a starting model: the data's generating model")
        ppo = prepare_ppo(model, data, args.batch_size, device, args.clip)
    else:
        if (data[:, OFF_POLICY_WEIGHT] < 0).any():
            raise SystemExit("imitation takes non-negative policy weights only")
        ppo = None
    names = head_names(ppo)

    action_class_weight = None
    if args.balance_actions > 0:
        labels = data[:, OFF_ACTION_TYPE]
        counts = np.array([(labels == c).sum() for c in range(NUM_ACTION_TYPES)], dtype=np.float64)
        total = counts.sum()
        # Inverse frequency to a power: 1 is fully balanced, lower powers lift the rare
        # MOVE label without letting the rarer DISBAND label dominate the head
        weights = np.where(counts > 0, (total / (NUM_ACTION_TYPES * np.maximum(counts, 1))) ** args.balance_actions, 1.0)
        action_class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
        print("action class weights: " + " ".join(f"{w:.2f}" for w in weights), flush=True)

    init_loss, init_heads = eval_loss(model, data, args.batch_size, device, action_class_weight, ppo)
    w = len(str(args.epochs))
    print(f"epoch {0:>{w}}/{args.epochs}: {_fmt(init_loss, init_heads, names)}", flush=True)

    for epoch in range(args.epochs):
        train_epoch(model, optimizer, data, args.batch_size, device, action_class_weight, ppo)
        avg_loss, heads = eval_loss(model, data, args.batch_size, device, action_class_weight, ppo)
        print(f"epoch {epoch + 1:>{w}}/{args.epochs}: {_fmt(avg_loss, heads, names)}", flush=True)

    ratios = " ".join(
        f"{name}={h / i:.2f}" if abs(i) > 1e-8 else f"{name}=N/A"
        for name, h, i in zip(names, heads, init_heads)
    )
    print(f"ratio: {ratios}", flush=True)

    model.cpu()
    export_model(model, args.model)
    print(f"loss={avg_loss:.8f}")


if __name__ == "__main__":
    main()
