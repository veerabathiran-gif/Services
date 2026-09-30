"""Train a small attention-based neural network (AbNN) on the 6 clinical features
selected by SSC-AE in the thesis, then export weights for the browser app.

Pipeline (mirrors thesis Steps 2, 4, 5):
  mean imputation (BMI) -> min-max scaling (glucose, BMI)
  -> per-factor embedding -> attention (tanh score, softmax over 6 factors)
  -> weighted features -> hidden ReLU layer -> sigmoid output
"""
import json
import autograd.numpy as np
import numpy as onp
import pandas as pd
from autograd import grad
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

SEED = 42
rng = onp.random.default_rng(SEED)

d = pd.read_csv("data.csv")
bmi_mean = float(d.bmi.mean())
d["bmi"] = d.bmi.fillna(bmi_mean)

WORK = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
SMOKE = ["never smoked", "formerly smoked", "smokes", "Unknown"]
G_MIN, G_MAX = float(d.avg_glucose_level.min()), float(d.avg_glucose_level.max())
B_MIN, B_MAX = float(d.bmi.min()), float(d.bmi.max())


def encode(df):
    return {
        "hyp": df.hypertension.values.astype(int),
        "heart": df.heart_disease.values.astype(int),
        "work": df.work_type.map({w: i for i, w in enumerate(WORK)}).values.astype(int),
        "smoke": df.smoking_status.map({s: i for i, s in enumerate(SMOKE)}).values.astype(int),
        "glu": ((df.avg_glucose_level.values - G_MIN) / (G_MAX - G_MIN)).astype(float),
        "bmi": ((df.bmi.values - B_MIN) / (B_MAX - B_MIN)).astype(float),
    }


idx = onp.arange(len(d))
tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=d.stroke, random_state=SEED)
tr2_idx, va_idx = train_test_split(tr_idx, test_size=0.15, stratify=d.stroke.values[tr_idx], random_state=SEED)
y = d.stroke.values.astype(float)

K, H = 4, 16
FACTORS = ["hyp", "heart", "work", "glu", "bmi", "smoke"]


def init():
    s = 0.3
    return {
        "E_hyp": rng.normal(0, s, (2, K)),
        "E_heart": rng.normal(0, s, (2, K)),
        "E_work": rng.normal(0, s, (len(WORK), K)),
        "E_smoke": rng.normal(0, s, (len(SMOKE), K)),
        "U_glu": rng.normal(0, s, K), "C_glu": rng.normal(0, s, K),
        "U_bmi": rng.normal(0, s, K), "C_bmi": rng.normal(0, s, K),
        "Wa": rng.normal(0, s, (K, K)), "ba": onp.zeros(K), "q": rng.normal(0, s, K),
        "W1": rng.normal(0, onp.sqrt(2 / (6 * K)), (6 * K, H)), "b1": onp.zeros(H),
        "w2": rng.normal(0, onp.sqrt(1 / H), H), "b2": onp.zeros(1),
    }


def forward(p, X, return_att=False):
    v = [
        p["E_hyp"][X["hyp"]],
        p["E_heart"][X["heart"]],
        p["E_work"][X["work"]],
        X["glu"][:, None] * p["U_glu"] + p["C_glu"],
        X["bmi"][:, None] * p["U_bmi"] + p["C_bmi"],
        p["E_smoke"][X["smoke"]],
    ]
    V = np.stack(v, axis=1)                                   # n x 6 x K
    e = np.dot(np.tanh(np.dot(V, p["Wa"]) + p["ba"]), p["q"])  # n x 6
    e = e - np.max(e, axis=1, keepdims=True)
    a = np.exp(e) / np.sum(np.exp(e), axis=1, keepdims=True)  # attention weights
    Z = (6.0 * a[:, :, None] * V).reshape((V.shape[0], 6 * K))  # weighted features
    h = np.maximum(0, np.dot(Z, p["W1"]) + p["b1"])            # hidden ReLU
    logit = np.dot(h, p["w2"]) + p["b2"][0]
    out = 1 / (1 + np.exp(-logit))                             # sigmoid
    return (out, a) if return_att else out


def subset(X, ids):
    return {k: v[ids] for k, v in X.items()}


Xall = encode(d)
Xtr, Xva, Xte = subset(Xall, tr2_idx), subset(Xall, va_idx), subset(Xall, te_idx)
ytr, yva, yte = y[tr2_idx], y[va_idx], y[te_idx]
pos_w = (ytr == 0).sum() / (ytr == 1).sum()


def loss(p, X, yy, l2=1e-3):
    o = np.clip(forward(p, X), 1e-7, 1 - 1e-7)
    bce = -np.mean(pos_w * yy * np.log(o) + (1 - yy) * np.log(1 - o)) / ((pos_w + 1) / 2)
    reg = sum(np.sum(v ** 2) for k, v in p.items() if k.startswith("W") or k == "w2")
    return bce + l2 * reg


g = grad(loss)
p = init()
m = {k: onp.zeros_like(v) for k, v in p.items()}
s2 = {k: onp.zeros_like(v) for k, v in p.items()}
lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
best, best_p, bad, t = -1, None, 0, 0
for epoch in range(400):
    perm = rng.permutation(len(ytr))
    for i in range(0, len(perm), 128):
        bi = perm[i:i + 128]
        gr = g(p, subset(Xtr, bi), ytr[bi])
        t += 1
        for k in p:
            m[k] = b1 * m[k] + (1 - b1) * gr[k]
            s2[k] = b2 * s2[k] + (1 - b2) * gr[k] ** 2
            p[k] = p[k] - lr * (m[k] / (1 - b1 ** t)) / (onp.sqrt(s2[k] / (1 - b2 ** t)) + eps)
    auc = roc_auc_score(yva, forward(p, Xva))
    if auc > best + 1e-4:
        best, best_p, bad = auc, {k: onp.array(v) for k, v in p.items()}, 0
    else:
        bad += 1
        if bad >= 30:
            break
print(f"stopped at epoch {epoch}, best val AUC {best:.3f}")
p = best_p

# threshold: maximise Youden's J on validation
pv = forward(p, Xva)
ths = onp.linspace(0.05, 0.95, 181)
J = [((pv >= th) & (yva == 1)).sum() / (yva == 1).sum() + ((pv < th) & (yva == 0)).sum() / (yva == 0).sum() - 1 for th in ths]
thr = float(ths[int(onp.argmax(J))])

pt, att = forward(p, Xte, return_att=True)
auc = roc_auc_score(yte, pt)
tn, fp, fn, tp = confusion_matrix(yte, (pt >= thr).astype(int)).ravel()
metrics = {
    "test_auc": round(float(auc), 3),
    "threshold": round(thr, 3),
    "sensitivity": round(tp / (tp + fn), 3),
    "specificity": round(tn / (tn + fp), 3),
    "accuracy": round((tp + tn) / (tp + tn + fp + fn), 3),
    "confusion": {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)},
    "n_total": int(len(d)), "n_stroke": int(y.sum()),
    "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
    "mean_attention": {f: round(float(att[:, i].mean()), 3) for i, f in enumerate(FACTORS)},
}
print(json.dumps(metrics, indent=1))

export = {
    "meta": {
        "WORK": WORK, "SMOKE": SMOKE, "FACTORS": FACTORS,
        "G_MIN": G_MIN, "G_MAX": G_MAX, "B_MIN": B_MIN, "B_MAX": B_MAX,
        "BMI_MEAN": round(bmi_mean, 2), "K": K, "H": H,
    },
    "metrics": metrics,
    "params": {k: onp.round(v, 6).tolist() for k, v in p.items()},
}
json.dump(export, open("model.json", "w"))

# reference predictions for JS parity check
cases = [
    dict(hyp=0, heart=0, work=0, smoke=0, glu=85.0, bmi=23.0),
    dict(hyp=1, heart=1, work=1, smoke=1, glu=228.7, bmi=36.6),
    dict(hyp=1, heart=0, work=0, smoke=2, glu=150.0, bmi=31.0),
    dict(hyp=0, heart=1, work=2, smoke=3, glu=105.0, bmi=bmi_mean),
    dict(hyp=0, heart=0, work=3, smoke=3, glu=95.0, bmi=18.0),
]
ref = []
for c in cases:
    X = {
        "hyp": onp.array([c["hyp"]]), "heart": onp.array([c["heart"]]),
        "work": onp.array([c["work"]]), "smoke": onp.array([c["smoke"]]),
        "glu": onp.array([(c["glu"] - G_MIN) / (G_MAX - G_MIN)]),
        "bmi": onp.array([(c["bmi"] - B_MIN) / (B_MAX - B_MIN)]),
    }
    o, a = forward(p, X, return_att=True)
    ref.append({"input": c, "score": float(o[0]), "att": [float(x) for x in a[0]]})
json.dump(ref, open("reference_cases.json", "w"), indent=1)
for r in ref:
    print(r["input"], round(r["score"], 4), [round(x, 3) for x in r["att"]])
