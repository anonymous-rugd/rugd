"""One-batch real-graph verification for modern models; this is not an experiment run."""
from __future__ import annotations

import argparse
import time

import torch

from traffic_ml.config import load_config
from traffic_ml.opencity_experiment import _to, loader, prepare
from traffic_ml.opencity_models import build_open_model

MODELS = ("rugd", "uignn_inductive", "staeformer_inductive", "opencity_scratch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify full-graph model memory and gradients.")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    cfg = load_config("configs/experiment.yaml")
    cfg["training"]["time_batch_size"] = args.batch_size
    stores, sources, _, datasets = prepare(cfg, "TrafficHZ")
    city = max(sources, key=lambda item: stores[item].adjacency.shape[0])
    batch = next(iter(loader(datasets[city, "validation"], cfg)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = _to(batch, device)
    adjacency = stores[city].normalized_adjacency().to(device)
    print(f"DEVICE | {device} | city={city} | nodes={adjacency.shape[0]}")
    for name in MODELS:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = build_open_model(name, cfg, len(sources)).to(device)
        model.train(args.backward)
        started = time.perf_counter()
        with torch.set_grad_enabled(args.backward), torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            output = model(batch, adjacency)
            loss = -output["logits"].mean()
        if args.backward:
            loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        parameters = sum(parameter.numel() for parameter in model.parameters())
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
        finite = bool(torch.isfinite(output["logits"]).all())
        print(f"VERIFY | {name:24s} | params={parameters:>9,d} | "
              f"forward={elapsed:7.3f}s | peak_cuda={peak:8.1f} MiB | finite={finite}")


if __name__ == "__main__":
    main()
