"""
Apply a model saved by main.py to new data. No retraining: the threshold and the
Gaussian statistics are loaded from model.json and used as-is.

Classify freshly generated signals (truth is known, so it reports accuracy):
    python predict.py --model experiments/<run>/model.json --mod 4ask --snr 0 --n_examples 5
    python predict.py --model experiments/<run>/model.json --mod 4psk --snr -4 --n_examples 500

Classify your own data saved as .npy (e.g. an SDR capture):
    python predict.py --model experiments/<run>/model.json --npy capture.npy
    Accepted shapes: complex (n,) or (num, n); float I/Q (2, n) or (num, 2, n).
    A long 1-D capture is cut into windows of the training length n.

Classify every example in a dataset saved by data_loader.py --out file.npz:
    python predict.py --model experiments/<run>/model.json --npz dataset.npz
"""

import argparse
import json

import numpy as np

from data_loader import make_example
from features import extract
from classifiers import ThresholdRule, GaussianBayes


def load_model(path):
    with open(path) as f:
        m = json.load(f)
    rule = ThresholdRule.from_dict(m["rule"], m["feature_names"], m["mods"])
    bayes = GaussianBayes.from_dict(m["bayes"], m["feature_names"], m["mods"])
    return m, rule, bayes


def load_npy(path, n):
    a = np.load(path)
    if not np.iscomplexobj(a):
        # float I/Q: (2, n) or (num, 2, n) -> complex
        if a.ndim == 2 and a.shape[0] == 2:
            a = a[0] + 1j * a[1]
        elif a.ndim == 3 and a.shape[1] == 2:
            a = a[:, 0] + 1j * a[:, 1]
        else:
            raise ValueError("Float data must be shaped (2, n) or (num, 2, n), got {}".format(a.shape))
    if a.ndim == 1 and len(a) > n:
        # one long capture: cut into non-overlapping windows of n samples
        k = len(a) // n
        print("Long capture of {} samples -> {} windows of {}".format(len(a), k, n))
        a = a[:k * n].reshape(k, n)
    return np.atleast_2d(a)


def report(X, mods, rule, bayes, truth=None):
    F = extract(X)
    rule_pred = rule.predict(F)
    probs = bayes.predict_proba(F)
    bayes_pred = bayes.classes[np.argmax(probs, axis=1)]

    show = min(len(X), 20)
    print("\n  #  | rule  | bayes | " + " | ".join("P({})".format(m.upper()) for m in mods)
          + (" | truth" if truth is not None else ""))
    for i in range(show):
        line = " {:>3d} | {:<5s} | {:<5s} | ".format(i, mods[rule_pred[i]].upper(), mods[bayes_pred[i]].upper())
        line += " | ".join("{:>7.3f}".format(p) for p in probs[i])
        if truth is not None:
            line += " | " + mods[truth[i]].upper()
        print(line)
    if len(X) > show:
        print("  ... ({} more)".format(len(X) - show))

    if truth is not None:
        print("\nAccuracy over {} examples: rule {:.4f}, bayes {:.4f}".format(
            len(X), (rule_pred == truth).mean(), (bayes_pred == truth).mean()))
    else:
        counts = {m.upper(): int((bayes_pred == k).sum()) for k, m in enumerate(mods)}
        print("\nBayes predictions over {} examples: {}".format(len(X), counts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to model.json from a main.py run")
    ap.add_argument("--npy", help="your own data as .npy")
    ap.add_argument("--npz", help="dataset saved by data_loader.py --out")
    ap.add_argument("--mod", help="generate new examples of this modulation, e.g. 4ask")
    ap.add_argument("--snr", type=float, default=0.0)
    ap.add_argument("--n_examples", type=int, default=5)
    ap.add_argument("--random_phase", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None, help="default: different examples every run")
    a = ap.parse_args()

    m, rule, bayes = load_model(a.model)
    mods, n = m["mods"], m["data"]["n"]
    print("Loaded model trained on {} at SNRs {}".format([x.upper() for x in mods], m["data"]["train_snrs"]))
    print("Rule: " + rule.describe())

    if a.npy:
        report(load_npy(a.npy, n), mods, rule, bayes)
    elif a.npz:
        d = np.load(a.npz)
        report(d["X"], mods, rule, bayes, truth=d["y"])
    elif a.mod:
        if a.mod not in mods:
            raise SystemExit("--mod must be one of {}".format(mods))
        rng = np.random.default_rng(a.seed)
        X = np.stack([make_example(rng, a.mod, a.snr, n=n, sps=m["data"]["sps"],
                                   random_phase=bool(a.random_phase)) for _ in range(a.n_examples)])
        print("Generated {} new {} examples at {} dB".format(a.n_examples, a.mod.upper(), a.snr))
        report(X, mods, rule, bayes, truth=np.full(a.n_examples, mods.index(a.mod)))
    else:
        raise SystemExit("Give one of --npy, --npz, or --mod")