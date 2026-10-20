# Spectrum-Sensing-ML-Algo

Feature-based (non-neural) modulation classification. Current task: **4PSK vs 4ASK**.

## How it works

1. **`data_loader.py`** generates complex baseband examples (128 samples, unit signal power)
   for any `M`-PSK or `M`-ASK at any list of SNRs, with an optional random carrier phase.
2. **`features.py`** turns each example into four interpretable numbers:

   | feature | formula | what it asks |
   |---|---|---|
   | `circ2` | \|mean(x²)\| / mean(\|x\|²) | Are all clusters on one line through the origin? (ASK ≈ SNR/(1+SNR), PSK ≈ 0) |
   | `circ4` | \|mean(x⁴)\| / mean(\|x\|²)² | Is there 4-fold phase symmetry? |
   | `amp_cv` | std(\|x\|) / mean(\|x\|) | Does the amplitude change? |
   | `kurt` | mean(\|x\|⁴) / mean(\|x\|²)² | Amplitude kurtosis |

   Squaring doubles every phase, so ASK's 0° and 180° both land on 0° and add up, while
   QPSK's 45°/135°/225°/315° land on ±90° and cancel. This works for any carrier phase.
3. **`classifiers.py`**
   - `ThresholdRule`: one feature, one learned threshold (a readable rule).
   - `GaussianBayes`: per-class Gaussian on all features (QDA, or naive Bayes with `--naive 1`),
     gives posterior probabilities.

## Run

```bash
python main.py                                             # train + test at 0 dB
python main.py --test_snrs -10 -6 -2 0 2 6 10              # sweep a 0 dB-trained model
python main.py --train_snrs -10 -6 -2 0 2 6 10 --test_snrs -10 -6 -2 0 2 6 10
python data_loader.py --snrs 0 10                          # constellation preview
```

The original BFSK / MLP spectrum-sensing code is kept in `legacy_nn/`.
