"""
Evaluate a trained spectrum-sensing model, broken down by SNR.

Run on a saved checkpoint:
    python test.py --ckpt_path experiments/<run_folder>/models/best.pth

Output:
    - prints a per-SNR table (accuracy, detection rate, false-alarm rate)
    - saves results.json and accuracy_vs_snr.png next to the checkpoint's run folder

solver.py also imports evaluate() and plot_results() from here, so the
numbers you see during training and at test time are computed the same way.
"""

import argparse
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")            # headless-safe (server / Colab)
import matplotlib.pyplot as plt

from model import MLPClassifier
from data_loader import get_loader


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Returns a dict with overall accuracy and, for every SNR:
        acc : fraction of all examples classified correctly
        pd  : probability of detection  = P(predict signal | signal present)
        pfa : probability of false alarm = P(predict signal | noise only)
    Pd and Pfa are the standard spectrum-sensing metrics.
    """
    model.eval()   # BatchNorm uses running stats, dropout off

    preds, labels, snrs = [], [], []
    for x, y, snr in loader:
        logits = model(x.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(y)
        snrs.append(snr)

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    snrs = torch.cat(snrs).numpy()

    per_snr = {}
    for s in sorted(set(snrs.tolist())):
        m = snrs == s
        p, y = preds[m], labels[m]
        per_snr[s] = {
            "acc": float((p == y).mean()),
            "pd": float((p[y == 1] == 1).mean()) if (y == 1).any() else float("nan"),
            "pfa": float((p[y == 0] == 1).mean()) if (y == 0).any() else float("nan"),
        }

    return {"overall_acc": float((preds == labels).mean()), "per_snr": per_snr}


def print_results(results):
    print("\n  SNR (dB) | accuracy |   Pd   |  Pfa")
    print("  ---------+----------+--------+-------")
    for s, r in results["per_snr"].items():
        print("  {:>8d} |  {:6.3f}  | {:6.3f} | {:5.3f}".format(s, r["acc"], r["pd"], r["pfa"]))
    print("\n  Overall accuracy: {:.4f}\n".format(results["overall_acc"]))


def plot_results(results, path):
    snrs = list(results["per_snr"].keys())
    acc = [results["per_snr"][s]["acc"] for s in snrs]
    pd = [results["per_snr"][s]["pd"] for s in snrs]
    pfa = [results["per_snr"][s]["pfa"] for s in snrs]

    plt.figure(figsize=(8, 5))
    plt.plot(snrs, acc, "o-", label="accuracy")
    plt.plot(snrs, pd, "s--", label="Pd (detection)")
    plt.plot(snrs, pfa, "^:", label="Pfa (false alarm)")
    plt.axhline(0.5, color="gray", lw=0.8, ls="--")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("SNR (dB)")
    plt.ylabel("rate")
    plt.title("Spectrum sensing performance vs SNR (overall acc {:.3f})".format(results["overall_acc"]))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = MLPClassifier(num_classes=cfg["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--test_n_per_snr", type=int, default=500)
    ap.add_argument("--test_seed", type=int, default=1)    # different from training seed (0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_dir", type=str, default="")      # default: <run_folder>/results
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(a.ckpt_path, device)

    # Test data uses the same signal settings the model was trained on
    test_loader = get_loader(batch_size=a.batch_size, mode="test",
                             n_per_snr=a.test_n_per_snr, h=cfg["h"],
                             max_cfo=cfg["cfo"], seed=a.test_seed)

    results = evaluate(model, test_loader, device)
    print_results(results)

    out_dir = a.out_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.ckpt_path))), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    plot_results(results, os.path.join(out_dir, "accuracy_vs_snr.png"))
    print("Saved results to", out_dir)