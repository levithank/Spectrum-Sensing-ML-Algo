
import argparse
import numpy as np

SPS = 8
N = 128
NBITS = 48
SNRS = list(range(-20, 20, 2))


def make_example(rng, snr_db=None, h=0.5, max_cfo=0.0):

    # Generate random binary bits (0 and 1)
    bits = rng.integers(0, 2, NBITS)

    # Convert bits to -1 and +1
    nrz = bits * 2 - 1

    # Repeat each bit for 8 samples
    freq = np.repeat(nrz, SPS).astype(float)

    # Generate phase from frequency
    phase = np.pi * h * np.cumsum(freq) / SPS

    # Generate complex IQ signal
    x = np.exp(1j * phase)

    # Select 128 consecutive samples
    start = rng.integers(0, len(x) - N + 1)
    x = x[start:start + N]

    # Add random carrier phase and optional frequency offset
    n = np.arange(N) 
    cfo = rng.uniform(-max_cfo, max_cfo) if max_cfo > 0 else 0.0

    x = x * np.exp(
        1j * (2 * np.pi * cfo * n + rng.uniform(0, 2 * np.pi))
    )

    # Add white Gaussian noise
    if snr_db is not None:
        noise_power = 10 ** (-snr_db / 10)
        noise = (
            rng.standard_normal(N)
            + 1j * rng.standard_normal(N)
        ) * np.sqrt(noise_power / 2)

        x = x + noise

    return x


def to_iq(x):
    return np.stack([x.real, x.imag]).astype(np.float32)


def build(n_per_snr, seed, h, max_cfo):

    rng = np.random.default_rng(seed)

    X, snr = [], []

    for s in SNRS:
        for _ in range(n_per_snr):
            x = make_example(rng, s, h, max_cfo)
            X.append(to_iq(x))
            snr.append(s)

    return np.stack(X), np.array(snr)


def plot_preview(path, h, seed=3):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [None, 18, 6]

    fig, axes = plt.subplots(3, 3, figsize=(13, 9))

    for j, s in enumerate(cols):

        rng = np.random.default_rng(seed)
        x = make_example(rng, s, h)

        label = "no noise" if s is None else f"{s} dB SNR"

        # Row 1: IQ versus time
        axes[0, j].plot(x.real, label="I")
        axes[0, j].plot(x.imag, "--", label="Q")
        axes[0, j].set_title(label)
        axes[0, j].set_xlabel("sample")

        # Row 2: Instantaneous frequency
        inst_f = np.angle(
            x[1:] * np.conj(x[:-1])
        ) / (2 * np.pi)

        axes[1, j].plot(inst_f)
        axes[1, j].set_ylim(-0.15, 0.15)
        axes[1, j].set_xlabel("sample")
        axes[1, j].set_ylabel("frequency (cycles/sample)")

        # Row 3: IQ constellation
        axes[2, j].scatter(x.real, x.imag, s=8)
        axes[2, j].set_aspect("equal")
        axes[2, j].set_xlim(-2.2, 2.2)
        axes[2, j].set_ylim(-2.2, 2.2)
        axes[2, j].set_xlabel("I")
        axes[2, j].set_ylabel("Q")

    axes[0, 0].legend()
    axes[0, 0].set_ylabel("amplitude")

    fig.suptitle(
        f"BFSK, h={h}, {SPS} samples per symbol",
        y=0.995
    )

    plt.tight_layout()
    plt.savefig(path, dpi=110)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--n-per-snr", type=int, default=500)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--cfo", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bfsk_dataset.npz")
    ap.add_argument("--plot", default="bfsk_preview.png")

    a = ap.parse_args()

    X, snr = build(a.n_per_snr, a.seed, a.h, a.cfo)

    np.savez_compressed(a.out, X=X, snr=snr)

    print("X:", X.shape, X.dtype)
    print("SNRs:", sorted(set(snr.tolist())))
    print("Saved to", a.out)

    if a.plot:
        plot_preview(a.plot, a.h)
        print("Plot saved to", a.plot)
