"""Create a random-weights model and export to TF.js format.

Usage:
  uv run python -m app.scripts.bootstrap --output path/to/model/
"""
import argparse
import numpy as np
import torch

from app.config import STATE_SIZE
from app.model import GraphConquestNN
from app.export_tfjs import export_model


def main():
    parser = argparse.ArgumentParser(description="Bootstrap random NN model")
    parser.add_argument("--output", required=True, help="Output directory for TF.js model")
    args = parser.parse_args()

    model = GraphConquestNN()

    # Quick sanity check
    x = torch.randn(1, STATE_SIZE)
    outputs = model(x)
    value = outputs[0].item()
    print(f"Sanity check — value: {value:.4f}")

    # Count params
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total}")

    export_model(model, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
