"""
Feature-based (non-neural) modulation classification. Starts with 4PSK vs 4ASK.

    python main.py                                    # train + test at 0 dB
    python main.py --test_snrs -10 -6 -2 0 2 6 10     # test the 0 dB model at other SNRs
    python main.py --train_snrs -10 -6 -2 0 2 6 10 --test_snrs -10 -6 -2 0 2 6 10

Results go to ./experiments/<run_name>_<time>/ :
    params.json, results/results.json, results/*.png
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np

from data_loader import build, plot_preview
from features import extract, FEATURE_NAMES, FEATURE_HELP
from classifiers import ThresholdRule, GaussianBayes
from eval import evaluate, print_results, plot_feature_hists, plot_scatter, plot_acc_vs_snr


def main(cfg):
    run = cfg.run_name.replace(" ", "_").replace("/", "_")
    t = datetime.now().strftime("%d%m_%H%M%S")
    shared = os.path.join(cfg.base_path, "{}_{}".format(run, t) if run else t)
    res_dir = os.path.join(shared, "results")
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(shared, "params.json"), "w") as f:
        json.dump(vars(cfg), f, indent=4, sort_keys=True)
    print("Run folder:", shared)

    mods = cfg.mods
    data_kw = dict(n=cfg.n, sps=cfg.sps, random_phase=bool(cfg.random_phase), max_cfo=cfg.cfo)

    # ---- data (different seeds so train and test never overlap)
    Xtr, ytr, str_ = build(mods, cfg.train_snrs, cfg.n_per_class, cfg.data_seed, **data_kw)
    Xte, yte, ste = build(mods, cfg.test_snrs, cfg.test_n_per_class, cfg.data_seed + 1, **data_kw)
    print("Train: {} examples at SNRs {} | Test: {} examples at SNRs {}".format(
        len(ytr), cfg.train_snrs, len(yte), cfg.test_snrs))

    # ---- features
    Ftr, Fte = extract(Xtr), extract(Xte)
    print("\nFeatures:")
    for n in FEATURE_NAMES:
        print("  {:<7s} {}".format(n, FEATURE_HELP[n]))

    # ---- classifiers
    rule = ThresholdRule(cfg.rule_feature, FEATURE_NAMES, mods).fit(Ftr, ytr)
    bayes = GaussianBayes(mods, FEATURE_NAMES, naive=bool(cfg.naive)).fit(Ftr, ytr)
    print("\nLearned rule:  " + rule.describe())
    print(bayes.describe())

    all_results = {}
    for name, clf in [("rule ({})".format(cfg.rule_feature), rule), ("gaussian bayes", bayes)]:
        r = evaluate(clf.predict(Fte), yte, ste, mods)
        print_results(name, r)
        all_results[name] = r

    # example posteriors from the probabilistic model
    p = bayes.predict_proba(Fte[:3])
    print("\nExample Bayes posteriors (first 3 test examples, truth = {}):".format(
        [mods[k].upper() for k in yte[:3]]))
    for row in p:
        print("  " + ", ".join("P({}) = {:.3f}".format(m.upper(), v) for m, v in zip(mods, row)))

    # ---- save the trained model so predict.py can reuse it on new data
    model = {"mods": mods, "feature_names": FEATURE_NAMES,
             "data": {"n": cfg.n, "sps": cfg.sps, "train_snrs": cfg.train_snrs},
             "rule": rule.to_dict(), "bayes": bayes.to_dict()}
    with open(os.path.join(shared, "model.json"), "w") as f:
        json.dump(model, f, indent=4)

    # ---- save results
    out = {"rule": rule.describe(), "rule_threshold": float(rule.threshold),
           "bayes_class_means": {mods[c]: dict(zip(FEATURE_NAMES, mu.tolist()))
                                 for c, mu in zip(bayes.classes, bayes.means)},
           "results": all_results}
    with open(os.path.join(res_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=4)
    plot_feature_hists(Fte, yte, FEATURE_NAMES, mods, os.path.join(res_dir, "feature_histograms.png"), rule)
    plot_scatter(Fte, yte, FEATURE_NAMES, mods, os.path.join(res_dir, "feature_scatter.png"), rule=rule)
    plot_preview(os.path.join(res_dir, "constellations.png"), mods, [None] + list(cfg.test_snrs)[:3])
    if len(cfg.test_snrs) > 1:
        plot_acc_vs_snr(all_results, os.path.join(res_dir, "accuracy_vs_snr.png"))
    print("\nSaved model to", os.path.join(shared, "model.json"))
    print("Saved results to", res_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--mods", nargs="+", default=["4psk", "4ask"])
    ap.add_argument("--train_snrs", nargs="+", type=int, default=[0])
    ap.add_argument("--test_snrs", nargs="+", type=int, default=[0])
    ap.add_argument("--n_per_class", type=int, default=2000)       # per SNR, per class
    ap.add_argument("--test_n_per_class", type=int, default=2000)
    ap.add_argument("--n", type=int, default=128)                  # samples per example
    ap.add_argument("--sps", type=int, default=1)                  # samples per symbol
    ap.add_argument("--random_phase", type=int, default=1)         # 1 = unknown carrier phase
    ap.add_argument("--cfo", type=float, default=0.0)              # max carrier freq offset
    ap.add_argument("--data_seed", type=int, default=0)            # train = seed, test = seed+1
    # classifiers
    ap.add_argument("--rule_feature", default="circ2", choices=FEATURE_NAMES)
    ap.add_argument("--naive", type=int, default=0)                # 1 = diagonal-covariance naive Bayes
    # misc
    ap.add_argument("--base_path", default="./experiments/")
    ap.add_argument("--run_name", default="")
    main(ap.parse_args())