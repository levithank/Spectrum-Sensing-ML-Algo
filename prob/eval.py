"""
Evaluation helpers: per-SNR accuracy, confusion matrix, and plots.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate(pred, y, snr, class_names):
    """Overall accuracy, per-SNR accuracy, and a confusion matrix (rows = truth, cols = prediction)."""
    k = len(class_names)
    conf = np.zeros((k, k), dtype=int)
    for t, p in zip(y, pred):
        conf[t, p] += 1
    per_snr = {int(s): float((pred[snr == s] == y[snr == s]).mean()) for s in sorted(set(snr.tolist()))}
    return {
        "overall_acc": float((pred == y).mean()),
        "per_snr_acc": per_snr,
        "confusion": conf.tolist(),
        "classes": list(class_names),
    }


def print_results(name, results):
    print("\n[{}] overall accuracy: {:.4f}".format(name, results["overall_acc"]))
    if len(results["per_snr_acc"]) > 1:
        print("  SNR (dB) | accuracy")
        for s, a in results["per_snr_acc"].items():
            print("  {:>8d} |  {:.4f}".format(s, a))
    cls = [c.upper() for c in results["classes"]]
    print("  confusion (rows = truth, cols = predicted):")
    print("  " + " " * 8 + "".join("{:>8s}".format(c) for c in cls))
    for c, row in zip(cls, results["confusion"]):
        print("  {:<8s}".format(c) + "".join("{:>8d}".format(v) for v in row))


def plot_feature_hists(F, y, feature_names, class_names, path, rule=None):
    fig, axes = plt.subplots(1, len(feature_names), figsize=(4 * len(feature_names), 3.4))
    for j, (ax, name) in enumerate(zip(np.atleast_1d(axes), feature_names)):
        lo, hi = np.percentile(F[:, j], [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        for c, cname in enumerate(class_names):
            ax.hist(F[y == c, j], bins=bins, alpha=0.55, label=cname.upper())
        if rule is not None and rule.feature == name:
            ax.axvline(rule.threshold, color="k", ls="--", lw=1.2, label="rule threshold")
        ax.set_title(name)
        ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_scatter(F, y, feature_names, class_names, path, fx="circ2", fy="amp_cv", rule=None):
    i, j = feature_names.index(fx), feature_names.index(fy)
    plt.figure(figsize=(5.5, 4.5))
    for c, cname in enumerate(class_names):
        plt.scatter(F[y == c, i], F[y == c, j], s=5, alpha=0.4, label=cname.upper())
    if rule is not None and rule.feature == fx:
        plt.axvline(rule.threshold, color="k", ls="--", lw=1.2, label="rule threshold")
    plt.xlabel(fx)
    plt.ylabel(fy)
    plt.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_acc_vs_snr(all_results, path):
    plt.figure(figsize=(7, 4.5))
    for name, r in all_results.items():
        s = list(r["per_snr_acc"].keys())
        plt.plot(s, [r["per_snr_acc"][k] for k in s], "o-", label=name)
    plt.ylim(0.45, 1.01)
    plt.xlabel("SNR (dB)")
    plt.ylabel("accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
