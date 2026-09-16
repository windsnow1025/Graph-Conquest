"""Train the model on binary sample data exported by TypeScript (v7, 10 heads).

Usage:
  uv run python -m app.scripts.train --data path/to/samples.bin --model path/to/model/
  uv run python -m app.scripts.train --data samples.bin --model public/model/ --epochs 10 --lr 0.001
"""
import argparse
import time

import torch

from app.model import GraphConquestNN
from app.data_io import read_samples, read_multiple
from app.trainer import train_epoch, eval_loss, HEAD_NAMES
from app.export_tfjs import export_model, import_tfjs_weights


def _fmt(total, heads):
    # Line 1: loss + non-battle heads (val act spl dis rec mov)
    line1_names = HEAD_NAMES[:6]  # val act spl dis rec mov
    line1 = " ".join(f"{name}={v:.6f}" for name, v in zip(line1_names, heads[:6]))
    # Line 2: battle heads (btgt bsel batk kfr ret)
    line2_names = HEAD_NAMES[6:]  # btgt bsel batk kfr ret
    line2 = " ".join(f"{name}={v:.6f}" for name, v in zip(line2_names, heads[6:]))
    return f"loss={total:.4f} {line1}\nbattle: {line2}"


def main():
    parser = argparse.ArgumentParser(description="Train Graph Conquest NN v5")
    parser.add_argument("--data", nargs="+", required=True, help="Binary sample files")
    parser.add_argument("--model", required=True, help="TF.js model directory (read + write)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fresh", action="store_true", help="Train from scratch (ignore existing weights)")
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

    _, init_heads = eval_loss(model, data, args.batch_size, device)
    w = len(str(args.epochs))
    print(f"epoch {0:>{w}}/{args.epochs}: {_fmt(sum(init_heads), init_heads)}", flush=True)

    for epoch in range(args.epochs):
        train_loss, train_heads = train_epoch(model, optimizer, data, args.batch_size, device)
        avg_loss, heads = eval_loss(model, data, args.batch_size, device)
        print(f"epoch {epoch + 1:>{w}}/{args.epochs}: {_fmt(avg_loss, heads)}", flush=True)

    ratios = " ".join(
        f"{name}={h / i:.2f}" if i > 1e-8 else f"{name}=N/A"
        for name, h, i in zip(HEAD_NAMES, heads, init_heads)
    )
    print(f"ratio: {ratios}", flush=True)

    model.cpu()
    export_model(model, args.model)
    print(f"loss={avg_loss:.8f}")


if __name__ == "__main__":
    main()
