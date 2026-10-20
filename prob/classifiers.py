"""
Two non-neural classifiers that work on the features from features.py.

ThresholdRule   - rule-based: one feature, one threshold. The threshold is the value
                  that best separates the training data (a decision stump), so the
                  learned model is a single readable line, e.g.
                  "if circ2 > 0.247 -> 4ASK else 4PSK".

GaussianBayes   - probabilistic: models each class's features as a Gaussian
                  (full covariance = QDA, or diagonal = naive Bayes) and returns
                  the posterior P(class | features) via Bayes' rule.

Both expose fit(F, y), predict(F), and describe(); GaussianBayes also has predict_proba(F).
"""

import numpy as np


class ThresholdRule:
    def __init__(self, feature, feature_names, class_names):
        self.feature = feature
        self.idx = feature_names.index(feature)
        self.class_names = class_names
        self.threshold = None
        self.high_class = None   # class predicted when feature > threshold

    def fit(self, F, y):
        if len(np.unique(y)) != 2:
            raise ValueError("ThresholdRule handles exactly two classes")
        f = F[:, self.idx]
        order = np.argsort(f)
        f_sorted, y_sorted = f[order], y[order]
        # candidate thresholds: midpoints between consecutive sorted values
        cands = (f_sorted[:-1] + f_sorted[1:]) / 2
        best = (-1, None, None)
        for c in (0, 1):
            # predict class c above threshold: accuracy = (#not-c below) + (#c above)
            is_c = (y_sorted == c).astype(int)
            below_not_c = np.cumsum(1 - is_c)[:-1]
            above_c = is_c[::-1].cumsum()[::-1][1:]
            acc = (below_not_c + above_c) / len(y)
            k = int(np.argmax(acc))
            if acc[k] > best[0]:
                best = (acc[k], cands[k], c)
        _, self.threshold, self.high_class = best
        return self

    def predict(self, F):
        above = F[:, self.idx] > self.threshold
        return np.where(above, self.high_class, 1 - self.high_class)

    def to_dict(self):
        return {"feature": self.feature, "threshold": float(self.threshold),
                "high_class": int(self.high_class)}

    @classmethod
    def from_dict(cls, d, feature_names, class_names):
        rule = cls(d["feature"], feature_names, class_names)
        rule.threshold = d["threshold"]
        rule.high_class = d["high_class"]
        return rule

    def describe(self):
        hi = self.class_names[self.high_class]
        lo = self.class_names[1 - self.high_class]
        return "if {} > {:.4f} -> {} else {}".format(self.feature, self.threshold, hi.upper(), lo.upper())


class GaussianBayes:
    def __init__(self, class_names, feature_names, naive=False, reg=1e-6):
        self.class_names = class_names
        self.feature_names = feature_names
        self.naive = naive
        self.reg = reg

    def fit(self, F, y):
        self.classes = np.unique(y)
        self.means, self.covs, self.log_priors = [], [], []
        for c in self.classes:
            Fc = F[y == c]
            mu = Fc.mean(axis=0)
            cov = np.atleast_2d(np.cov(Fc, rowvar=False))   # 1 feature -> 1x1 matrix, not a scalar
            if self.naive:
                cov = np.diag(np.diag(cov))
            cov = cov + self.reg * np.eye(F.shape[1])
            self.means.append(mu)
            self.covs.append(cov)
            self.log_priors.append(np.log(len(Fc) / len(F)))
        return self

    def _log_joint(self, F):
        out = []
        for mu, cov, lp in zip(self.means, self.covs, self.log_priors):
            d = F - mu
            inv = np.linalg.inv(cov)
            _, logdet = np.linalg.slogdet(cov)
            maha = np.einsum("ij,jk,ik->i", d, inv, d)
            out.append(lp - 0.5 * (maha + logdet + F.shape[1] * np.log(2 * np.pi)))
        return np.stack(out, axis=1)

    def predict_proba(self, F):
        lj = self._log_joint(F)
        lj -= lj.max(axis=1, keepdims=True)
        p = np.exp(lj)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, F):
        return self.classes[np.argmax(self._log_joint(F), axis=1)]

    def to_dict(self):
        return {"naive": self.naive, "reg": self.reg,
                "classes": self.classes.tolist(),
                "means": [m.tolist() for m in self.means],
                "covs": [c.tolist() for c in self.covs],
                "log_priors": [float(p) for p in self.log_priors]}

    @classmethod
    def from_dict(cls, d, feature_names, class_names):
        model = cls(class_names, feature_names, naive=d["naive"], reg=d["reg"])
        model.classes = np.array(d["classes"])
        model.means = [np.array(m) for m in d["means"]]
        model.covs = [np.array(c) for c in d["covs"]]
        model.log_priors = d["log_priors"]
        return model

    def describe(self):
        lines = ["Gaussian {} (per-class feature means):".format("naive Bayes" if self.naive else "Bayes / QDA")]
        lines.append(" " * 10 + "".join("{:>10s}".format(n) for n in self.feature_names))
        for c, mu in zip(self.classes, self.means):
            lines.append("  {:<8s}".format(self.class_names[c].upper())
                         + "".join("{:>10.4f}".format(v) for v in mu))
        return "\n".join(lines)