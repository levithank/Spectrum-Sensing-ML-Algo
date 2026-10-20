"""
Synthetic modulated-signal generator for feature-based modulation classification.

Currently supports M-PSK and M-ASK (bipolar / PAM). Every example is a complex
baseband window of N samples with unit average signal power, optionally rotated
by a random carrier phase, plus complex white Gaussian noise at a chosen SNR.

Nothing here assumes a particular SNR: pass any list of SNRs (in dB) to build().

    from data_loader import build
    X, y, snr = build(["4psk", "4ask"], snrs=[0], n_per_class=1000, seed=0)
    # X   : complex64 [num_examples, N]
    # y   : int64     [num_examples]   index into the mods list
    # snr : int64     [num_examples]

Run directly to save a constellation preview:
    python data_loader.py --snrs 0 10 30
"""

import argparse
import re

import numpy as np

N = 128          # samples per example
SPS = 1          # samples per symbol (1 = one sample per symbol, i.e. 128 symbols)


# --------------------------------------------------------------------------
# Constellations (all normalised to unit average power)
# --------------------------------------------------------------------------

def psk_constellation(M):
    """M points on the unit circle, spaced 360/M degrees apart.
    QPSK gets the usual 45-degree offset (points at 45, 135, 225, 315)."""
    offset = np.pi / 4 if M == 4 else 0.0
    return np.exp(1j * (2 * np.pi * np.arange(M) / M + offset))


def ask_constellation(M):
    """Bipolar M-ASK (M-PAM): levels -(M-1), ..., -1, +1, ..., +(M-1) on the real axis.
    Every point has phase 0 or 180 degrees, so all clusters lie on one line."""
    levels = 2 * np.arange(M) - (M - 1)
    pts = levels.astype(complex)
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


CONSTELLATIONS = {
    "psk": psk_constellation,
    "ask": ask_constellation,
}


def parse_mod(name):
    """'4psk' -> ('psk', 4). Also accepts 'qpsk' and 'bpsk'."""
    name = name.lower().replace("-", "")
    aliases = {"bpsk": "2psk", "qpsk": "4psk"}
    name = aliases.get(name, name)
    m = re.fullmatch(r"(\d+)([a-z]+)", name)
    if not m or m.group(2) not in CONSTELLATIONS:
        raise ValueError("Unknown modulation '{}'. Use e.g. 4psk, 8psk, 4ask.".format(name))
    return m.group(2), int(m.group(1))


# --------------------------------------------------------------------------
# Signal + noise
# --------------------------------------------------------------------------

def make_noise(rng, snr_db, n=N):
    """Complex AWGN. Signal power is 1, so noise power = 10^(-SNR/10)."""
    noise_power = 10 ** (-snr_db / 10)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(noise_power / 2)


def make_example(rng, mod, snr_db=None, n=N, sps=SPS, random_phase=True, max_cfo=0.0):
    """
    One example of modulation `mod` (e.g. '4psk').

    snr_db       : None for a noiseless signal, otherwise SNR in dB
    sps          : samples per symbol (rectangular pulses)
    random_phase : rotate by a random carrier phase, as an unsynchronised receiver sees it.
                   With it off, 4ASK clusters sit exactly on the I axis.
    max_cfo      : max carrier frequency offset in cycles/sample (0 = none)
    """
    kind, M = parse_mod(mod)
    pts = CONSTELLATIONS[kind](M)

    n_sym = int(np.ceil(n / sps)) + 1
    symbols = pts[rng.integers(0, M, n_sym)]
    x = np.repeat(symbols, sps)

    start = rng.integers(0, sps)            # random symbol timing
    x = x[start:start + n]

    k = np.arange(n)
    phase0 = rng.uniform(0, 2 * np.pi) if random_phase else 0.0
    cfo = rng.uniform(-max_cfo, max_cfo) if max_cfo > 0 else 0.0
    x = x * np.exp(1j * (2 * np.pi * cfo * k + phase0))

    if snr_db is not None:
        x = x + make_noise(rng, snr_db, n)
    return x.astype(np.complex64)


def build(mods, snrs=(0,), n_per_class=1000, seed=0, n=N, sps=SPS,
          random_phase=True, max_cfo=0.0):
    """
    For every SNR and every modulation, generate n_per_class examples.

    Returns:
        X   : complex64 [num_examples, n]
        y   : int64     [num_examples]   label = index of the modulation in `mods`
        snr : int64     [num_examples]
    """
    rng = np.random.default_rng(seed)
    X, y, snr = [], [], []
    for s in snrs:
        for label, mod in enumerate(mods):
            for _ in range(n_per_class):
                X.append(make_example(rng, mod, s, n, sps, random_phase, max_cfo))
                y.append(label)
                snr.append(s)
    return np.stack(X), np.array(y, dtype=np.int64), np.array(snr, dtype=np.int64)


def to_iq(x):
    """complex [..., n] -> float32 [..., 2, n] (I row, Q row), the old NN input format."""
    return np.stack([x.real, x.imag], axis=-2).astype(np.float32)


# --------------------------------------------------------------------------
# Preview
# --------------------------------------------------------------------------

def plot_preview(path, mods=("4psk", "4ask"), snrs=(None, 10, 0), seed=3, random_phase=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(mods), len(snrs), figsize=(3.6 * len(snrs), 3.6 * len(mods)),
                             squeeze=False)
    for i, mod in enumerate(mods):
        for j, s in enumerate(snrs):
            rng = np.random.default_rng(seed)
            x = make_example(rng, mod, s, n=1024, random_phase=random_phase)
            ax = axes[i, j]
            ax.scatter(x.real, x.imag, s=4, alpha=0.5)
            ax.axhline(0, color="gray", lw=0.6)
            ax.axvline(0, color="gray", lw=0.6)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect("equal")
            ax.set_title("{}, {}".format(mod.upper(), "no noise" if s is None else "{} dB".format(s)))
    fig.suptitle("Constellations (random carrier phase {})".format("on" if random_phase else "off"))
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mods", nargs="+", default=["4psk", "4ask"])
    ap.add_argument("--snrs", nargs="+", type=int, default=[0])
    ap.add_argument("--n_per_class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random_phase", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--plot", default="constellation_preview.png")
    a = ap.parse_args()

    X, y, snr = build(a.mods, a.snrs, a.n_per_class, a.seed, random_phase=bool(a.random_phase))
    print("X:", X.shape, X.dtype, "| classes:", dict(enumerate(a.mods)), "| SNRs:", sorted(set(snr.tolist())))
    if a.out:
        np.savez_compressed(a.out, X=X, y=y, snr=snr, mods=np.array(a.mods))
        print("Saved to", a.out)
    if a.plot:
        plot_preview(a.plot, a.mods, [None] + a.snrs)
        print("Plot saved to", a.plot)
