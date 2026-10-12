"""
Feed individual samples into a trained classifier and see its prediction.

Examples:
    # 5 random samples (random signal/noise, random SNR)
    python predict.py --ckpt_path experiments/<run_folder>/models/best.pth --n 5

    # one BFSK signal at -10 dB, with a plot of what the model saw
    python predict.py --ckpt_path ... --kind signal --snr -10 --plot

    # one noise-only sample at 0 dB
    python predict.py --ckpt_path ... --kind noise --snr 0

    # your own sample: a .npy file holding either a float array of shape (2, 128)
    # [I row, Q row] or a complex array of shape (128,)
    python predict.py --ckpt_path ... --npy my_sample.npy
"""

import argparse

import numpy as np
import torch

from model import MLPClassifier
from data_loader import make_example, make_noise, to_iq, SNRS, N

CLASS_NAMES = {0: "noise only", 1: "signal present"}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = MLPClassifier(num_classes=cfg["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()   # important: BatchNorm uses its saved statistics, so a single sample works
    return model, cfg


@torch.no_grad()
def predict(model, iq, device):
    """iq: float32 array of shape (2, 128). Returns (predicted class, probabilities)."""
    x = torch.from_numpy(iq).unsqueeze(0).to(device)   # add batch dimension -> (1, 2, 128)
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return int(probs.argmax()), probs


def load_npy(path):
    a = np.load(path)
    if np.iscomplexobj(a):
        a = to_iq(a)
    a = np.asarray(a, dtype=np.float32)
    if a.shape != (2, N):
        raise ValueError("Expected shape (2, {0}) or complex ({0},), got {1}".format(N, a.shape))
    return a


def plot_sample(iq, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.plot(iq[0], label="I")
    ax1.plot(iq[1], "--", label="Q")
    ax1.set_xlabel("sample")
    ax1.set_ylabel("amplitude")
    ax1.legend()
    ax2.scatter(iq[0], iq[1], s=8)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_xlabel("I")
    ax2.set_ylabel("Q")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--kind", choices=["random", "signal", "noise"], default="random")
    ap.add_argument("--snr", type=int, default=None)       # default: random SNR from the training range
    ap.add_argument("--n", type=int, default=1)            # how many samples to try
    ap.add_argument("--seed", type=int, default=None)      # default: different samples every run
    ap.add_argument("--npy", type=str, default=None)       # classify your own saved sample instead
    ap.add_argument("--plot", action="store_true")         # save a picture of each sample
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(a.ckpt_path, device)

    # ---- your own sample
    if a.npy:
        iq = load_npy(a.npy)
        pred, probs = predict(model, iq, device)
        print("\nFile: {}".format(a.npy))
        print("Prediction: {}  (P(noise only) = {:.3f}, P(signal present) = {:.3f})\n".format(
            CLASS_NAMES[pred], probs[0], probs[1]))
        if a.plot:
            plot_sample(iq, "Prediction: " + CLASS_NAMES[pred], "prediction_sample.png")
            print("Plot saved to prediction_sample.png")
        raise SystemExit

    # ---- freshly generated samples (same signal settings the model was trained on)
    rng = np.random.default_rng(a.seed)
    correct = 0

    print()
    for i in range(a.n):
        snr = a.snr if a.snr is not None else int(rng.choice(SNRS))
        kind = a.kind if a.kind != "random" else rng.choice(["signal", "noise"])

        if kind == "signal":
            iq = to_iq(make_example(rng, snr, cfg["h"], cfg["cfo"]))
            true_label = 1
        else:
            iq = to_iq(make_noise(rng, snr))
            true_label = 0

        pred, probs = predict(model, iq, device)
        ok = pred == true_label
        correct += ok

        print("Sample {:>3d} | SNR {:>4d} dB | truth: {:<14s} | predicted: {:<14s} | "
              "P(signal) = {:.3f}  {}".format(i + 1, snr, CLASS_NAMES[true_label],
                                               CLASS_NAMES[pred], probs[1],
                                               "correct" if ok else "WRONG"))

        if a.plot:
            path = "prediction_sample_{}.png".format(i + 1)
            plot_sample(iq, "SNR {} dB | truth: {} | predicted: {} (P(signal) = {:.2f})".format(
                snr, CLASS_NAMES[true_label], CLASS_NAMES[pred], probs[1]), path)

    print("\n{}/{} correct".format(correct, a.n))
    if a.plot:
        print("Plots saved as prediction_sample_<n>.png in the current folder")