# %% [markdown]
# # CSET485 AI and Society – Assignment 1 (COMPAS)
# **Roll number:** S24CSEU1192 → dataset **COMPAS** (ProPublica `compas-scores-two-years.csv`), `random_state = 192`, 70/30 train/test split.
#
# Every number cited in `report.pdf` is produced by this notebook and saved to `results.json`; figures go to `figures/`.

# %%
import os, json, urllib.request, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")
SEED = 192
URL = "https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv"
CSV = "data/compas-scores-two-years.csv"
os.makedirs("data", exist_ok=True); os.makedirs("figures", exist_ok=True)
R = {}  # results store
pd.set_option("display.width", 160)

# %% [markdown]
# ## 0. Data loading
# The CSV is downloaded from ProPublica's GitHub if it is not already present; a copy is committed in `data/` for reproducibility.

# %%
if not os.path.exists(CSV):
    urllib.request.urlretrieve(URL, CSV)
raw = pd.read_csv(CSV)
R["n_raw"] = int(len(raw)); R["n_cols_raw"] = int(raw.shape[1])
print(raw.shape)

# %% [markdown]
# ## 1. Preprocessing
# ### 1.1 Missing values
# We document missingness in the raw file (many columns relate to violent-recidivism events and are legitimately empty).

# %%
miss = raw.isna().sum(); miss = miss[miss > 0].sort_values(ascending=False)
R["missing_raw"] = {k: int(v) for k, v in miss.items()}
print(miss)

# %% [markdown]
# ### 1.2 ProPublica filters
# * `days_b_screening_arrest` within ±30: if the arrest and COMPAS screening are more than 30 days apart, we may not have the right offence matched to the score (rows with NaN here are also dropped, as in ProPublica).
# * `is_recid != -1`: −1 means no COMPAS case was found – the outcome is unknown.
# * `c_charge_degree != 'O'`: ordinary traffic offences do not result in jail time and are excluded.
# * `score_text != 'N/A'`: rows without a usable COMPAS score.

# %%
steps = {}
df = raw.copy(); steps["raw"] = len(df)
df = df[(df.days_b_screening_arrest <= 30) & (df.days_b_screening_arrest >= -30)]; steps["days_b_screening_arrest in ±30"] = len(df)
df = df[df.is_recid != -1]; steps["is_recid != -1"] = len(df)
df = df[df.c_charge_degree != "O"]; steps["c_charge_degree != 'O'"] = len(df)
df = df[df.score_text != "N/A"]; steps["score_text != 'N/A'"] = len(df)
R["filter_steps"] = {k: int(v) for k, v in steps.items()}; R["n_filtered"] = int(len(df))
print(steps)

# %% [markdown]
# ### 1.3 Features, encoding, outliers, scaling
# Features (identical for both models, **no score columns**): `age`, `priors_count`, `juv_fel_count`, `juv_misd_count`, `juv_other_count`, `sex`, `race`, `c_charge_degree`.
# * Categorical encoding: one-hot with a dropped reference (sex: Female; race: Caucasian; charge: M). Race groups Asian and Native American (tiny counts) are merged into "Other".
# * Outliers: count variables are heavily right-skewed; we report how many rows exceed Q3+1.5·IQR but **keep** them (they are real criminal histories, and removing them would bias toward low-risk people); no capping.
# * Scaling: numeric features are standardised with a `StandardScaler` fitted on the training set only, so coefficients are "per 1 SD".
# * After filtering, the selected columns have no missing values.

# %%
num = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]
df["race_g"] = df.race.replace({"Asian": "Other", "Native American": "Other"})
R["missing_after"] = int(df[num + ["sex", "race", "c_charge_degree", "decile_score", "two_year_recid"]].isna().sum().sum())
out = {}
for c in num:
    q1, q3 = df[c].quantile([.25, .75]); out[c] = int((df[c] > q3 + 1.5 * (q3 - q1)).sum())
R["outliers_iqr"] = out
R["race_counts"] = {k: int(v) for k, v in df.race_g.value_counts().items()}
R["sex_counts"] = {k: int(v) for k, v in df.sex.value_counts().items()}
R["base_rate"] = float(df.two_year_recid.mean())
X_cat = pd.DataFrame({"sex_Male": (df.sex == "Male").astype(float),
                      "race_AfricanAmerican": (df.race_g == "African-American").astype(float),
                      "race_Hispanic": (df.race_g == "Hispanic").astype(float),
                      "race_Other": (df.race_g == "Other").astype(float),
                      "charge_Felony": (df.c_charge_degree == "F").astype(float)}, index=df.index)
X_all = pd.concat([df[num].astype(float), X_cat], axis=1)
FEATS = list(X_all.columns)
print(out, R["missing_after"], FEATS)
print(df[num].describe().T)

# %%
idx_tr, idx_te = train_test_split(df.index, test_size=0.30, random_state=SEED)
scaler = StandardScaler().fit(X_all.loc[idx_tr, num])
def prep(idx):
    X = X_all.loc[idx].copy(); X[num] = scaler.transform(X[num]); return sm.add_constant(X, has_constant="add")
Xtr, Xte = prep(idx_tr), prep(idx_te)
R["priors_mean"], R["priors_sd"] = float(scaler.mean_[1]), float(scaler.scale_[1])
R["n_train"], R["n_test"] = int(len(idx_tr)), int(len(idx_te))
print(len(idx_tr), len(idx_te))

# %% [markdown]
# ## 2. Part 1 – Model A (OLS on `decile_score`) vs Model B (logistic on `two_year_recid`)
# `two_year_recid` is the **"is_recidivist"** outcome: 1 if the person was re-arrested within two years.
# Model A is thresholded at its median test prediction (≥ median → high risk); Model B at p ≥ 0.5.

# %%
mA = sm.OLS(df.loc[idx_tr, "decile_score"], Xtr).fit()
mB = sm.Logit(df.loc[idx_tr, "two_year_recid"], Xtr).fit(disp=0)
predA = mA.predict(Xte); pB = mB.predict(Xte)
medA = float(np.median(predA))
hA = (predA >= medA).astype(int); hB = (pB >= 0.5).astype(int)
cm = confusion_matrix(hA, hB, labels=[0, 1])
mism = int((hA != hB).sum())
yte = df.loc[idx_te, "two_year_recid"]
R["part1"] = dict(median_predA=medA, predA_min=float(predA.min()), predA_max=float(predA.max()),
    olsR2=float(mA.rsquared), cm=cm.tolist(), mismatch=mism, mismatch_pct=100 * mism / len(hA),
    highA=int(hA.sum()), highB=int(hB.sum()), highA_pct=100*hA.mean(), highB_pct=100*hB.mean(),
    accA=float((hA == yte).mean()), accB=float((hB == yte).mean()),
    pB_median=float(np.median(pB)), pB_at_medA=None,
    coefA={k: float(v) for k, v in mA.params.items()}, coefB={k: float(v) for k, v in mB.params.items()},
    corr_AB=float(np.corrcoef(predA, pB)[0, 1]))
# share of test with pB>=0.5 vs implied threshold
R["part1"]["pB_quantile_used"] = float((pB < 0.5).mean())
print(pd.DataFrame(cm, index=["A low", "A high"], columns=["B low", "B high"]))
print(json.dumps({k: v for k, v in R["part1"].items() if not k.startswith("coef")}, indent=1))
print(pd.DataFrame({"OLS (decile)": mA.params, "Logit (log-odds)": mB.params}).round(3))

# %% [markdown]
# ## 3. Part 2.1 – Simpson's paradox check
# Logistic regression of `two_year_recid` on the full (filtered) data, then separately for African-American and Caucasian defendants (race dummies dropped inside subgroups). We report coefficients, 95% CIs and p-values, and test the difference of each coefficient between groups with a z-test.

# %%
from scipy import stats
Xfull = X_all.copy(); Xfull[num] = scaler.transform(Xfull[num]); Xfull = sm.add_constant(Xfull)
y = df.two_year_recid
def fit(mask, cols):
    return sm.Logit(y[mask], Xfull.loc[mask, ["const"] + cols]).fit(disp=0)
base_cols = [c for c in FEATS]
sub_cols = [c for c in FEATS if not c.startswith("race_")]
m_full = fit(df.index == df.index, base_cols)
mAA = fit(df.race_g == "African-American", sub_cols)
mCA = fit(df.race_g == "Caucasian", sub_cols)
rows = []
for c in sub_cols:
    for name, m in [("Full", m_full), ("African-American", mAA), ("Caucasian", mCA)]:
        ci = m.conf_int().loc[c]
        rows.append(dict(predictor=c, model=name, coef=m.params[c], ci_lo=ci[0], ci_hi=ci[1], p=m.pvalues[c], n=int(m.nobs)))
simp = pd.DataFrame(rows)
diff = []
for c in sub_cols:
    d = mAA.params[c] - mCA.params[c]; se = np.sqrt(mAA.bse[c]**2 + mCA.bse[c]**2); z = d / se
    diff.append(dict(predictor=c, diff=d, z=z, p=2 * (1 - stats.norm.cdf(abs(z))),
                     sign_reversal=bool(np.sign(mAA.params[c]) != np.sign(mCA.params[c])),
                     full_sign=np.sign(m_full.params[c])))
diff = pd.DataFrame(diff)
R["simpson"] = dict(table=simp.to_dict("records"), diff=diff.to_dict("records"),
                    race_coef_full=dict(coef=float(m_full.params["race_AfricanAmerican"]), p=float(m_full.pvalues["race_AfricanAmerican"]),
                                        ci=[float(x) for x in m_full.conf_int().loc["race_AfricanAmerican"]]))
print(simp.round(4).to_string()); print(diff.round(4).to_string())

# %% [markdown]
# Sex-based check as a second split (Male vs Female subgroups), to see whether any sign reversal appears there.

# %%
cols_s = [c for c in FEATS if c != "sex_Male"]
mM = fit(df.sex == "Male", cols_s); mF = fit(df.sex == "Female", cols_s)
sx = []
for c in cols_s:
    d = mM.params[c] - mF.params[c]; se = np.sqrt(mM.bse[c]**2 + mF.bse[c]**2)
    cM, cF, cA = mM.conf_int().loc[c], mF.conf_int().loc[c], m_full.conf_int().loc[c]
    sx.append(dict(predictor=c, full=m_full.params[c], full_lo=cA[0], full_hi=cA[1], full_p=m_full.pvalues[c],
                   male=mM.params[c], male_lo=cM[0], male_hi=cM[1], male_p=mM.pvalues[c],
                   female=mF.params[c], female_lo=cF[0], female_hi=cF[1], female_p=mF.pvalues[c],
                   z=d/se, p_diff=2*(1-stats.norm.cdf(abs(d/se))), reversal=bool(np.sign(mM.params[c]) != np.sign(mF.params[c]))))
sx = pd.DataFrame(sx); R["n_male"], R["n_female"] = int(mM.nobs), int(mF.nobs)
R["aa_share_male"] = float((df[df.sex=="Male"].race_g=="African-American").mean()); R["aa_share_female"] = float((df[df.sex=="Female"].race_g=="African-American").mean())
R["recid_by_sex_race"] = {f"{a}|{b}": float(v) for (a, b), v in df[df.race_g.isin(["African-American","Caucasian"])].groupby(["sex","race_g"]).two_year_recid.mean().items()}
R["simpson_sex"] = sx.to_dict("records"); print(sx.round(4).to_string())

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
preds = sub_cols; yy = np.arange(len(preds))
for k, (name, m, col) in enumerate([("Full", m_full, "gray"), ("African-American", mAA, "tab:red"), ("Caucasian", mCA, "tab:blue")]):
    ci = m.conf_int().loc[preds]
    ax.errorbar(m.params[preds], yy + (k - 1) * 0.22, xerr=[m.params[preds] - ci[0], ci[1] - m.params[preds]], fmt="o", color=col, label=name, capsize=3)
ax.axvline(0, color="k", lw=.8); ax.set_yticks(yy); ax.set_yticklabels(preds); ax.set_xlabel("log-odds coefficient (95% CI); numeric = per 1 SD")
ax.set_title("2.1 Subgroup logistic coefficients"); ax.legend(); fig.tight_layout(); fig.savefig("figures/fig_simpson.png", dpi=150); plt.close()

# %% [markdown]
# ## 4. Part 2.2 – Omitted-variable bias
# Drop the confounder `priors_count` (correlated with age, race and charge degree, and the strongest predictor of recidivism) from the training-set logistic model and compare coefficients: % change = (reduced − full)/|full| × 100.

# %%
red_cols = [c for c in FEATS if c != "priors_count"]
mRed = sm.Logit(df.loc[idx_tr, "two_year_recid"], Xtr[["const"] + red_cols]).fit(disp=0)
ovb = pd.DataFrame({"full": mB.params[red_cols], "reduced": mRed.params[red_cols]})
ovb["abs_change"] = ovb.reduced - ovb.full
ovb["pct_chg"] = 100 * ovb.abs_change / ovb.full.abs()
ovb["full_p"] = mB.pvalues[red_cols]; ovb["reduced_p"] = mRed.pvalues[red_cols]
big = ovb.pct_chg.abs().idxmax()
corr = df[["priors_count"]].join(X_all[red_cols]).corr()["priors_count"].drop("priors_count")
R["ovb"] = dict(table=ovb.reset_index().rename(columns={"index": "predictor"}).to_dict("records"), biggest=big,
                biggest_pct=float(ovb.loc[big, "pct_chg"]), corr_with_priors={k: float(v) for k, v in corr.items()},
                llf_full=float(mB.llf), llf_red=float(mRed.llf), aic_full=float(mB.aic), aic_red=float(mRed.aic),
                priors_coef=float(mB.params["priors_count"]))
print(ovb.round(4).to_string()); print("biggest:", big); print(corr.round(3))

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ovb.pct_chg.plot.barh(ax=ax, color=["tab:red" if i == big else "tab:gray" for i in ovb.index])
ax.axvline(0, color="k", lw=.8); ax.set_xlabel("% change in coefficient after dropping priors_count"); ax.set_title("2.2 Omitted-variable bias")
fig.tight_layout(); fig.savefig("figures/fig_ovb.png", dpi=150); plt.close()

# %% [markdown]
# ## 5. Part 2.3 – Adversarial (confidently wrong) subset
# Test rows where Model B is confident and wrong: p ≥ 0.75 with y = 0, or p ≤ 0.25 with y = 1. We sample 50 with seed 192 and compare demographics with the full test set.

# %%
te = df.loc[idx_te].copy(); te["p"] = pB; te["y"] = yte
adv = te[((te.p >= 0.75) & (te.y == 0)) | ((te.p <= 0.25) & (te.y == 1))]
adv_hi = int(((te.p >= 0.75) & (te.y == 0)).sum()); adv_lo = int(((te.p <= 0.25) & (te.y == 1)).sum())
s50 = adv.sample(n=min(50, len(adv)), random_state=SEED)
s50.to_csv("data/adversarial_sample50.csv")
def demo(d):
    return {"n": len(d), "% African-American": 100*(d.race_g == "African-American").mean(), "% Caucasian": 100*(d.race_g == "Caucasian").mean(),
            "% Male": 100*(d.sex == "Male").mean(), "mean age": d.age.mean(), "% age<25": 100*(d.age < 25).mean(),
            "% age>45": 100*(d.age > 45).mean(), "mean priors": d.priors_count.mean(), "% priors=0": 100*(d.priors_count == 0).mean(),
            "% priors>=5": 100*(d.priors_count >= 5).mean(), "% felony": 100*(d.c_charge_degree == "F").mean(),
            "% y=1 (recidivated)": 100*d.y.mean(), "mean p": d.p.mean()}
demo_t = pd.DataFrame({"Full test": demo(te), "Adversarial (all)": demo(adv), "Sample of 50": demo(s50)})
s_hi = s50[s50.p >= 0.75]; s_lo = s50[s50.p <= 0.25]
traits = {"sample_hi_n": len(s_hi), "sample_lo_n": len(s_lo),
          "hi_pct_priors_ge5": 100*(s_hi.priors_count >= 5).mean() if len(s_hi) else None,
          "hi_pct_age_lt25": 100*(s_hi.age < 25).mean() if len(s_hi) else None,
          "lo_pct_priors0": 100*(s_lo.priors_count == 0).mean() if len(s_lo) else None,
          "lo_pct_age_gt45": 100*(s_lo.age > 45).mean() if len(s_lo) else None,
          "lo_pct_age_ge35": 100*(s_lo.age >= 35).mean() if len(s_lo) else None,
          "pct_extreme_profile": 100*(((s50.priors_count >= 5) | (s50.age < 25)) & (s50.p >= .75) | ((s50.priors_count <= 1) & (s50.age >= 35)) & (s50.p <= .25)).mean()}
R["adv"] = dict(count=int(len(adv)), pct=100*len(adv)/len(te), n_hi=adv_hi, n_lo=adv_lo,
                demo={c: {k: float(v) for k, v in demo_t[c].items()} for c in demo_t.columns}, traits={k: (None if v is None else float(v)) for k, v in traits.items()})
print(len(adv), 100*len(adv)/len(te), adv_hi, adv_lo); print(demo_t.round(2)); print(traits)

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].hist(te.priors_count.clip(upper=20), bins=21, alpha=.5, density=True, label="Full test")
axs[0].hist(s50.priors_count.clip(upper=20), bins=21, alpha=.6, density=True, label="Adversarial 50")
axs[0].set_xlabel("priors_count (clipped at 20)"); axs[0].legend()
axs[1].scatter(te.age, te.p, s=4, c="lightgray", label="test")
axs[1].scatter(s50.age, s50.p, s=20, c=np.where(s50.y == 1, "tab:red", "tab:blue"), label="adv 50 (red: y=1, blue: y=0)")
axs[1].set_xlabel("age"); axs[1].set_ylabel("Model B p"); axs[1].legend(fontsize=7)
fig.suptitle("2.3 Confidently-wrong subset"); fig.tight_layout(); fig.savefig("figures/fig_adversarial.png", dpi=150); plt.close()

# %% [markdown]
# ## 6. Part 2.4 – Fairness threshold sweep (African-American = A, Caucasian = B)
# DP = P(ŷ=1|A)/P(ŷ=1|B); EO = min(TPR_A/TPR_B, FPR_A/FPR_B); PP = precision_A/precision_B, on the test set with Model B's probabilities. Accuracy is computed on the whole test set.

# %%
def rates(d, t):
    yh = (d.p >= t).astype(int); y_ = d.y
    tp = ((yh == 1) & (y_ == 1)).sum(); fp = ((yh == 1) & (y_ == 0)).sum()
    return dict(sel=yh.mean(), tpr=tp / max((y_ == 1).sum(), 1), fpr=fp / max((y_ == 0).sum(), 1), prec=tp / (tp + fp) if tp + fp else np.nan)
A_ = te[te.race_g == "African-American"]; B_ = te[te.race_g == "Caucasian"]
sw = []
for t in np.round(np.arange(0.05, 0.96, 0.1), 2):
    a, b = rates(A_, t), rates(B_, t)
    sw.append(dict(t=t, DP=a["sel"]/b["sel"] if b["sel"] else np.nan,
                   EO=min(a["tpr"]/b["tpr"] if b["tpr"] else np.nan, a["fpr"]/b["fpr"] if b["fpr"] else np.nan),
                   PP=a["prec"]/b["prec"] if b["prec"] == b["prec"] and b["prec"] else np.nan,
                   acc=((te.p >= t).astype(int) == te.y).mean(), acc_A=((A_.p >= t).astype(int) == A_.y).mean(), acc_B=((B_.p >= t).astype(int) == B_.y).mean(),
                   sel_A=a["sel"], sel_B=b["sel"], TPR_A=a["tpr"], TPR_B=b["tpr"], FPR_A=a["fpr"], FPR_B=b["fpr"], prec_A=a["prec"], prec_B=b["prec"]))
sw = pd.DataFrame(sw)
R["sweep"] = sw.replace({np.nan: None}).to_dict("records"); R["n_test_AA"] = int(len(A_)); R["n_test_CA"] = int(len(B_))
R["base_A"] = float(A_.y.mean()); R["base_B"] = float(B_.y.mean())
print(sw.round(3).to_string())

# %% [markdown]
# **Trade-off quantification.** Baseline is t = 0.5. Fairness of a ratio r is measured by closeness to 1, f(r) = min(r, 1/r) ∈ (0, 1]. We pick the threshold (0.15–0.85, where both groups have positives) with the best DP closeness, and report: "improve DP from X to Y (Z% gain), EO drops from A to B (W% loss), cost = W/Z" where Z, W are % changes of the closeness scores.

# %%
close = lambda r: np.nan if r is None or r != r or r == 0 else min(r, 1/r)
sw["DPc"] = sw.DP.map(close); sw["EOc"] = sw.EO.map(close)
base = sw.loc[(sw.t - 0.45).abs() < 1e-9].iloc[0] if False else None
# t grid has 0.45 and 0.55, not 0.5 -> compute baseline at exactly 0.5 separately
a, b = rates(A_, 0.5), rates(B_, 0.5)
b05 = dict(acc_A=((A_.p >= .5) == A_.y).mean(), acc_B=((B_.p >= .5) == B_.y).mean(), DP=a["sel"]/b["sel"], EO=min(a["tpr"]/b["tpr"], a["fpr"]/b["fpr"]), PP=a["prec"]/b["prec"], acc=((te.p >= .5) == te.y).mean(),
           sel_A=a["sel"], sel_B=b["sel"], TPR_A=a["tpr"], TPR_B=b["tpr"], FPR_A=a["fpr"], FPR_B=b["fpr"], prec_A=a["prec"], prec_B=b["prec"])
b05["DPc"], b05["EOc"] = close(b05["DP"]), close(b05["EO"])
cand = sw[(sw.t >= 0.15) & (sw.t <= 0.85)]
best = cand.loc[cand.DPc.idxmax()]
Z = 100*(best.DPc - b05["DPc"]) / b05["DPc"]; W = 100*(b05["EOc"] - best.EOc) / b05["EOc"]
R["baseline_05"] = {k: float(v) for k, v in b05.items()}
R["tradeoff"] = dict(t_best=float(best.t), DP_from=b05["DP"], DP_to=float(best.DP), DPc_from=b05["DPc"], DPc_to=float(best.DPc), Z=float(Z),
                     EO_from=b05["EO"], EO_to=float(best.EO), EOc_from=b05["EOc"], EOc_to=float(best.EOc), W=float(W), cost=float(W/Z) if Z else None,
                     acc_from=float(b05["acc"]), acc_to=float(best.acc))
def tradeoff(row):
    d = dict(t=float(row.t), DP_from=b05["DP"], DP_to=float(row.DP), Z=100*(row.DPc-b05["DPc"])/b05["DPc"],
             EO_from=b05["EO"], EO_to=float(row.EO), W=100*(b05["EOc"]-row.EOc)/b05["EOc"],
             PP_from=b05["PP"], PP_to=float(row.PP), W_pp=100*(close(b05["PP"])-close(row.PP))/close(b05["PP"]),
             acc_from=float(b05["acc"]), acc_to=float(row.acc), W_acc=100*(b05["acc"]-row.acc)/b05["acc"])
    d["cost"] = d["W"]/d["Z"]; d["cost_pp"] = d["W_pp"]/d["Z"]; d["cost_acc"] = d["W_acc"]/d["Z"]; return {k: float(v) for k, v in d.items()}
R["tradeoffs"] = {f"{t:.2f}": tradeoff(sw.loc[(sw.t-t).abs()<1e-9].iloc[0]) for t in [0.15, 0.25, 0.35, 0.45]}
R["acc_best_t"] = float(sw.loc[sw.acc.idxmax(), "t"])
print(b05); print(R["tradeoff"]); print(pd.DataFrame(R["tradeoffs"]).round(3))

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
off = []
for k, ax in enumerate(axs):
    ax.plot(sw.DP, sw.EO, "-o", color="tab:purple")
    for n, r in sw.iterrows():
        inside = 0 <= r.DP <= 1.5 and 0 <= r.EO <= 1.5
        if k == 1 or inside:
            ax.annotate(f"t={r.t:.2f}", (r.DP, r.EO), textcoords="offset points", xytext=(6, -12 + 9 * (n % 3)), fontsize=8)
        elif k == 0: off.append(f"t={r.t:.2f} ({r.DP:.2f}, {r.EO:.2f})")
    ax.plot([1], [1], "k*", ms=12, label="perfect parity (1,1)"); ax.axvline(1, ls=":", c="gray"); ax.axhline(1, ls=":", c="gray")
    ax.set_xlabel("Demographic parity ratio DP (AA / Caucasian)"); ax.set_ylabel("Equalised-odds ratio EO")
axs[0].set_xlim(0, 1.5); axs[0].set_ylim(0, 1.5); axs[0].set_title("DP vs EO, axes 0-1.5 (required view)")
axs[0].text(0.05, 0.05, "Outside axes:\n" + "\n".join(off), fontsize=7.5, transform=axs[0].transAxes, va="bottom", bbox=dict(fc="w", ec="gray"))
axs[0].legend(loc="upper left"); axs[1].set_title("Full range (context)")
fig.suptitle("2.4 DP vs EO across thresholds (labels = threshold t)"); fig.tight_layout(); fig.savefig("figures/fig_dp_eo.png", dpi=150); plt.close()
R["dp_eo_off_axes"] = off
fig, ax = plt.subplots(figsize=(7.5, 4))
for c, l in [("TPR_A", "TPR AA"), ("TPR_B", "TPR Cauc"), ("FPR_A", "FPR AA"), ("FPR_B", "FPR Cauc"), ("acc", "Accuracy (all)")]:
    ax.plot(sw.t, sw[c], "-o", label=l, ms=4)
ax.set_xlabel("threshold t"); ax.set_ylabel("rate"); ax.legend(fontsize=8); ax.set_title("2.4 Group-specific rates and accuracy")
fig.tight_layout(); fig.savefig("figures/fig_rates.png", dpi=150); plt.close()

# %% [markdown]
# ## 7. Persist all results
# `results.json` is the single source for every number in the PDF report (built by `build_report.py`).

# %%
def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_): return bool(o)
    return str(o)
with open("results.json", "w") as f: json.dump(R, f, indent=1, default=conv)
print("saved results.json; keys:", list(R))
