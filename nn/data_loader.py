import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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
        x = x + make_noise(rng, snr_db)

    return x


def make_noise(rng, snr_db):
    # Complex white Gaussian noise at the power that corresponds to this SNR
    # (signal power is 1, so noise power = 10^(-SNR/10)).
    # Used alone, this is a "no signal present" example (label 0).
    noise_power = 10 ** (-snr_db / 10)
    return (
        rng.standard_normal(N)
        + 1j * rng.standard_normal(N)
    ) * np.sqrt(noise_power / 2)


def to_iq(x):
    return np.stack([x.real, x.imag]).astype(np.float32)


def build(n_per_snr, seed, h, max_cfo):
    """
    For every SNR, generate n_per_snr signal+noise examples (label 1)
    and n_per_snr noise-only examples (label 0) at the same noise level.

    Returns:
        X   : float32 [num_examples, 2, 128]  (I and Q channels)
        y   : int64   [num_examples]          (1 = signal present, 0 = noise only)
        snr : int64   [num_examples]          (SNR the example was generated at)
    """
    rng = np.random.default_rng(seed)

    X, y, snr = [], [], []

    for s in SNRS:
        for _ in range(n_per_snr):
            # Signal present
            X.append(to_iq(make_example(rng, s, h, max_cfo)))
            y.append(1)
            snr.append(s)

            # Noise only
            X.append(to_iq(make_noise(rng, s)))
            y.append(0)
            snr.append(s)

    return np.stack(X), np.array(y, dtype=np.int64), np.array(snr, dtype=np.int64)


class SpectrumDataset(Dataset):
    """
    Builds the dataset in memory (no .npz file needed), the same way the
    reference repo's Dataset classes generate their data in __init__.
    """

    def __init__(self, n_per_snr=500, seed=0, h=0.5, max_cfo=0.0):
        X, y, snr = build(n_per_snr, seed, h, max_cfo)
        self.X = torch.from_numpy(X)      # [num_examples, 2, 128]
        self.y = torch.from_numpy(y)      # [num_examples]
        self.snr = torch.from_numpy(snr)  # [num_examples]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # SNR is returned too so test.py can compute accuracy per SNR
        return self.X[idx], self.y[idx], self.snr[idx]


def get_loader(batch_size=64, mode='train', num_workers=0,
               n_per_snr=500, h=0.5, max_cfo=0.0, seed=None):
    """
    mode='train' -> seed 0 by default, shuffled, drops the last partial batch
                    (BatchNorm in the model fails on a batch of size 1)
    mode='test'  -> seed 1 by default, so test data never overlaps training data
    """
    if seed is None:
        seed = 0 if mode == 'train' else 1

    dataset = SpectrumDataset(n_per_snr=n_per_snr, seed=seed, h=h, max_cfo=max_cfo)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=(mode == 'train'),
                      drop_last=(mode == 'train'),
                      num_workers=num_workers)


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
    # Optional: still lets you save the dataset to .npz and make the preview plot.
    # Training does NOT need this; get_loader() builds the data in memory.

    ap = argparse.ArgumentParser()

    ap.add_argument("--n-per-snr", type=int, default=500)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--cfo", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bfsk_dataset.npz")
    ap.add_argument("--plot", default="bfsk_preview.png")

    a = ap.parse_args()

    X, y, snr = build(a.n_per_snr, a.seed, a.h, a.cfo)

    np.savez_compressed(a.out, X=X, y=y, snr=snr)

    print("X:", X.shape, X.dtype)
    print("y: signal =", int(y.sum()), " noise only =", int((y == 0).sum()))
    print("SNRs:", sorted(set(snr.tolist())))
    print("Saved to", a.out)

    if a.plot:
        plot_preview(a.plot, a.h)
        print("Plot saved to", a.plot)