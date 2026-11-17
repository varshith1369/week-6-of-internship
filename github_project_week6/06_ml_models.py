"""Week 5 - Machine Learning: regression (price) + classification supplement (high-value diamonds)
Run:  python ml_models.py        (needs diamonds_clean.csv from Week 2)
Outputs: ml_log.txt, model_results.csv, tuning_results_*.csv, figures/m*.png
"""
import time, warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
sns.set_style("whitegrid")
BLUE, GREEN, RED, ORANGE, GREY, PURPLE = "#2b6cb0", "#2f855a", "#c53030", "#dd6b20", "#718096", "#805ad5"
log = []; T0 = time.time()
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)
SEED = 42
CUT = ["Fair", "Good", "Very Good", "Premium", "Ideal"]; COLOR = list("DEFGHIJ"); CLARITY = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
df = pd.read_csv("diamonds_clean.csv")
ROUND = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
c2 = df.carat.round(2)
df["round_ct"] = c2.isin(ROUND).astype(int)
df["just_below"] = c2.isin([round(r - d, 2) for r in ROUND for d in (0.01, 0.02)]).astype(int)
df["color_ord"] = df.color.map({k: 6 - i for i, k in enumerate(COLOR)}); df["clarity_ord"] = df.clarity.map({k: i for i, k in enumerate(CLARITY)}); df["cut_ord"] = df.cut.map({k: i for i, k in enumerate(CUT)})
say("rows", len(df), "| round_ct share %.1f%% | just_below share %.1f%%" % (df.round_ct.mean() * 100, df.just_below.mean() * 100))
A = ["log_carat", "depth", "table", "cut_ord", "color_ord", "clarity_ord"]; Bf = A + ["round_ct", "just_below"]
NUM_L = ["log_carat", "depth", "table"]; NUM_LB = NUM_L + ["round_ct", "just_below"]; CAT = ["cut", "color", "clarity"]
y = df.log_price
idx_tr, idx_te = train_test_split(df.index, test_size=0.2, random_state=SEED)
tr, te = df.loc[idx_tr], df.loc[idx_te]; ytr, yte = y.loc[idx_tr], y.loc[idx_te]
say("train", len(tr), "test", len(te), "| test price range $%d-$%d" % (te.price.min(), te.price.max()))
kf = KFold(5, shuffle=True, random_state=SEED)
def lin_pre(num): return ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(drop="first"), CAT)])
def metrics(name, pred, yt=yte):
    act = np.exp(yt); prd = np.exp(pred); ape = np.abs(prd - act) / act
    return dict(model=name, R2_log=r2_score(yt, pred), RMSE_log=np.sqrt(mean_squared_error(yt, pred)), MAE_usd=mean_absolute_error(act, prd), RMSE_usd=np.sqrt(mean_squared_error(act, prd)), median_APE_pct=np.median(ape) * 100, MAPE_pct=ape.mean() * 100)
res = []; preds = {}

# ---- baselines
m = DummyRegressor().fit(tr[A], ytr); preds["Mean baseline"] = m.predict(te[A]); res.append(metrics("Mean baseline", preds["Mean baseline"]))
m = LinearRegression().fit(tr[["log_carat"]], ytr); preds["Linear: carat only"] = m.predict(te[["log_carat"]]); res.append(metrics("Linear: carat only", preds["Linear: carat only"]))
pipe = Pipeline([("pre", lin_pre(NUM_L)), ("m", LinearRegression())]).fit(tr, ytr); preds["Linear: carat + grades"] = pipe.predict(te); res.append(metrics("Linear: carat + grades", preds["Linear: carat + grades"]))
cv_lin = -cross_val_score(Pipeline([("pre", lin_pre(NUM_L)), ("m", LinearRegression())]), tr, ytr, cv=kf, scoring="neg_root_mean_squared_error"); say("Linear carat+grades 5-fold CV RMSE(log): %.4f +/- %.4f" % (cv_lin.mean(), cv_lin.std()))
cvs = {"Linear: carat + grades": cv_lin.mean()}

# ---- Ridge (grid search) with threshold features
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
gs = GridSearchCV(Pipeline([("pre", lin_pre(NUM_LB)), ("m", Ridge())]), {"m__alpha": alphas}, cv=kf, scoring="neg_root_mean_squared_error", return_train_score=True).fit(tr, ytr)
ridge_res = pd.DataFrame(gs.cv_results_)[["param_m__alpha", "mean_test_score", "std_test_score", "mean_train_score"]]; ridge_res.to_csv("tuning_results_ridge.csv", index=False)
say("Ridge grid best alpha", gs.best_params_, "CV RMSE %.4f" % -gs.best_score_)
preds["Ridge (tuned)"] = gs.predict(te); res.append(metrics("Ridge (tuned)", preds["Ridge (tuned)"])); cvs["Ridge (tuned)"] = -gs.best_score_

# ---- Random forest (randomized search on 12k subsample, then refit)
sub = tr.sample(12000, random_state=SEED); ysub = ytr.loc[sub.index]
rf_space = {"n_estimators": [100, 200], "max_depth": [8, 12, 16, 24, None], "min_samples_leaf": [1, 2, 5, 10], "max_features": [0.4, 0.6, 0.8, 1.0]}
t = time.time(); rs = RandomizedSearchCV(RandomForestRegressor(random_state=SEED, n_jobs=1), rf_space, n_iter=8, cv=3, scoring="neg_root_mean_squared_error", random_state=SEED).fit(sub[Bf], ysub)
pd.DataFrame(rs.cv_results_)[["param_n_estimators", "param_max_depth", "param_min_samples_leaf", "param_max_features", "mean_test_score", "rank_test_score"]].to_csv("tuning_results_rf.csv", index=False)
say("RF search done in %.0fs best %s CV RMSE %.4f" % (time.time() - t, rs.best_params_, -rs.best_score_))
t = time.time(); rf = RandomForestRegressor(**rs.best_params_, random_state=SEED, n_jobs=1).fit(tr[Bf], ytr); say("RF refit %.0fs" % (time.time() - t))
preds["Random forest (tuned)"] = rf.predict(te[Bf]); res.append(metrics("Random forest (tuned)", preds["Random forest (tuned)"])); cvs["Random forest (tuned)"] = -rs.best_score_
rf_cv_df = pd.DataFrame(rs.cv_results_)

# ---- Gradient boosting (randomized search)
hgb_space = {"learning_rate": [0.03, 0.05, 0.1, 0.2], "max_iter": [200, 300, 500], "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [10, 20, 50], "l2_regularization": [0, 0.1, 1.0]}
hsub = tr.sample(15000, random_state=SEED); hysub = ytr.loc[hsub.index]
t = time.time(); hs = RandomizedSearchCV(HistGradientBoostingRegressor(random_state=SEED), hgb_space, n_iter=10, cv=3, scoring="neg_root_mean_squared_error", random_state=SEED).fit(hsub[Bf], hysub)
hgb_cv_df = pd.DataFrame(hs.cv_results_); hgb_cv_df[["param_learning_rate", "param_max_iter", "param_max_leaf_nodes", "param_min_samples_leaf", "param_l2_regularization", "mean_test_score", "rank_test_score"]].to_csv("tuning_results_hgb.csv", index=False)
say("HGB search done in %.0fs best %s CV RMSE %.4f" % (time.time() - t, hs.best_params_, -hs.best_score_))
hgb = HistGradientBoostingRegressor(**hs.best_params_, random_state=SEED).fit(tr[Bf], ytr)
preds["Gradient boosting (tuned)"] = hgb.predict(te[Bf]); res.append(metrics("Gradient boosting (tuned)", preds["Gradient boosting (tuned)"])); cvs["Gradient boosting (tuned)"] = -hs.best_score_
hd = HistGradientBoostingRegressor(random_state=SEED).fit(tr[Bf], ytr); preds["Gradient boosting (default)"] = hd.predict(te[Bf]); res.append(metrics("Gradient boosting (default)", preds["Gradient boosting (default)"]))
R = pd.DataFrame(res).round(4); R["CV_RMSE_log"] = R.model.map(cvs).round(4); R.to_csv("model_results.csv", index=False); say("\nTEST RESULTS\n" + R.to_string(index=False))
best_name = R[R.model.str.contains("tuned")].sort_values("RMSE_log").iloc[0].model; say("best model:", best_name)
bestpred = preds[best_name]

# ---- 5-fold CV for best model (stability)
cvh = -cross_val_score(HistGradientBoostingRegressor(**hs.best_params_, random_state=SEED), tr[Bf], ytr, cv=kf, scoring="neg_root_mean_squared_error"); say("HGB tuned 5-fold CV RMSE(log): %.4f +/- %.4f" % (cvh.mean(), cvh.std()))
# paired comparison on test: squared errors HGB vs linear
e_l = (yte - preds["Linear: carat + grades"]) ** 2; e_h = (yte - preds["Gradient boosting (tuned)"]) ** 2; w = stats.wilcoxon(e_l, e_h); say("Wilcoxon paired squared errors linear vs HGB p=%.3g; HGB lower in %.1f%% of test rows" % (w.pvalue, (e_h < e_l).mean() * 100))
boots = []; rng = np.random.default_rng(SEED); n_te = len(te); ev = np.exp(yte.values); pv = np.exp(preds["Gradient boosting (tuned)"])
for _ in range(500):
    i = rng.integers(0, n_te, n_te); boots.append(np.mean(np.abs(pv[i] - ev[i])))
say("bootstrap 95%% CI for test MAE($) HGB: [%.1f, %.1f]" % tuple(np.percentile(boots, [2.5, 97.5])))

# ---- ablation: feature set A vs B
ab = []
for nm, mk, fa, fb in [("Gradient boosting", lambda: HistGradientBoostingRegressor(**hs.best_params_, random_state=SEED), A, Bf), ("Ridge", None, None, None)]:
    if mk is not None:
        ma = mk().fit(tr[fa], ytr); mb = mk().fit(tr[fb], ytr); ab.append((nm, "A: 6 base features", np.sqrt(mean_squared_error(yte, ma.predict(te[fa])))))
        ab.append((nm, "B: + round-weight features", np.sqrt(mean_squared_error(yte, mb.predict(te[fb])))))
    else:
        for lab, num in [("A: 6 base features", NUM_L), ("B: + round-weight features", NUM_LB)]:
            p_ = Pipeline([("pre", lin_pre(num)), ("m", Ridge(alpha=gs.best_params_["m__alpha"]))]).fit(tr, ytr); ab.append((nm, lab, np.sqrt(mean_squared_error(yte, p_.predict(te)))))
ab = pd.DataFrame(ab, columns=["model", "features", "RMSE_log"]); say("\nABLATION\n" + ab.round(4).to_string(index=False))

# ---- permutation importance
pi = permutation_importance(hgb, te[Bf], yte, n_repeats=5, random_state=SEED, scoring="neg_root_mean_squared_error"); imp = pd.Series(pi.importances_mean, index=Bf).sort_values(); say("permutation importance (RMSE increase):", imp.round(4).to_dict())

# ---- learning curve
sizes, trs, cvs_ = learning_curve(HistGradientBoostingRegressor(**{**hs.best_params_, "max_iter": min(hs.best_params_["max_iter"], 300)}, random_state=SEED), tr[Bf], ytr, train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], cv=3, scoring="neg_root_mean_squared_error")
say("learning curve sizes", sizes.tolist(), "CV RMSE", (-cvs_.mean(axis=1)).round(4).tolist(), "train RMSE", (-trs.mean(axis=1)).round(4).tolist())

# ---- error analysis by price decile
te2 = te.copy(); te2["act"] = np.exp(yte); te2["ape_best"] = np.abs(np.exp(bestpred) - te2.act) / te2.act * 100; te2["ape_lin"] = np.abs(np.exp(preds["Linear: carat + grades"]) - te2.act) / te2.act * 100
te2["dec"] = pd.qcut(te2.act, 10, labels=False) + 1; dec = te2.groupby("dec").agg(price_lo=("act", "min"), price_hi=("act", "max"), best=("ape_best", "median"), lin=("ape_lin", "median")).round(1); say("\nmedian APE by price decile:\n" + dec.to_string())
resid = yte - bestpred; say("test residuals (log): mean %.4f sd %.4f skew %.2f | within +/-10%% price error: %.1f%% | within +/-20%%: %.1f%%" % (resid.mean(), resid.std(), stats.skew(resid), (te2.ape_best <= 10).mean() * 100, (te2.ape_best <= 20).mean() * 100))
big = te2.sort_values("ape_best", ascending=False).head(5)[["carat", "cut", "color", "clarity", "act", "ape_best"]]; say("5 largest % errors:\n" + big.round(1).to_string(index=False))

# ===== FIGURES =====
order = ["Mean baseline", "Linear: carat only", "Linear: carat + grades", "Ridge (tuned)", "Random forest (tuned)", "Gradient boosting (tuned)"]
Rm = R.set_index("model").loc[order]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); cols = [GREY, GREY, BLUE, BLUE, GREEN, GREEN]
ax[0].barh(order[::-1], Rm.RMSE_log[::-1], color=cols[::-1]); ax[0].set_xlabel("Test RMSE of ln(price) (lower is better)"); ax[0].set_title("Model comparison: error on ln(price)")
for i, v in enumerate(Rm.RMSE_log[::-1]): ax[0].text(v + .005, i, f"{v:.3f}", va="center", fontsize=8)
ax[1].barh(order[::-1], Rm.MAE_usd[::-1], color=cols[::-1]); ax[1].set_xlabel("Test mean absolute error (USD)"); ax[1].set_title("Model comparison: error in dollars"); ax[1].set_yticklabels([])
for i, v in enumerate(Rm.MAE_usd[::-1]): ax[1].text(v + 15, i, f"${v:,.0f}", va="center", fontsize=8)
plt.tight_layout(); plt.savefig("figures/m1_model_comparison.png"); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
ax[0].errorbar(np.log10(ridge_res.param_m__alpha.astype(float)), -ridge_res.mean_test_score, yerr=ridge_res.std_test_score, marker="o", color=BLUE, capsize=3, label="CV"); ax[0].plot(np.log10(ridge_res.param_m__alpha.astype(float)), -ridge_res.mean_train_score, "s--", color=GREY, label="Train")
ax[0].set_xlabel("log10(alpha)"); ax[0].set_ylabel("RMSE of ln(price)"); ax[0].set_title("Ridge: grid search"); ax[0].legend(fontsize=8)
hd_ = hgb_cv_df.copy(); hd_["rmse"] = -hd_.mean_test_score; sc = ax[1].scatter(hd_.param_learning_rate.astype(float), hd_.rmse, c=hd_.param_max_leaf_nodes.astype(int), s=hd_.param_max_iter.astype(int) / 6, cmap="viridis", edgecolor="k", lw=.3)
ax[1].set_xscale("log"); ax[1].set_xlabel("learning_rate"); ax[1].set_ylabel("CV RMSE of ln(price)"); ax[1].set_title("Gradient boosting: random search"); plt.colorbar(sc, ax=ax[1], label="max_leaf_nodes")
rd_ = rf_cv_df.copy(); rd_["rmse"] = -rd_.mean_test_score; rd_["depth"] = rd_.param_max_depth.fillna(32).astype(float); sc2 = ax[2].scatter(rd_.depth, rd_.rmse, c=rd_.param_min_samples_leaf.astype(int), s=rd_.param_max_features.astype(float) * 150, cmap="plasma", edgecolor="k", lw=.3)
ax[2].set_xlabel("max_depth (32 = None)"); ax[2].set_ylabel("CV RMSE of ln(price)"); ax[2].set_title("Random forest: random search"); plt.colorbar(sc2, ax=ax[2], label="min_samples_leaf")
plt.tight_layout(); plt.savefig("figures/m2_tuning.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); act = np.exp(yte); prd = np.exp(bestpred)
h = ax[0].hexbin(act, prd, gridsize=55, xscale="log", yscale="log", bins="log", cmap="viridis", mincnt=1); lim = [act.min(), act.max()]; ax[0].plot(lim, lim, color=RED, ls="--"); ax[0].set_xlabel("Actual price (USD)"); ax[0].set_ylabel("Predicted price (USD)"); ax[0].set_title(f"{best_name}: test set"); plt.colorbar(h, ax=ax[0], label="Count (log)")
pe_ = (prd - act) / act * 100; sns.histplot(pe_[np.abs(pe_) < 60], bins=60, ax=ax[1], color=GREEN); ax[1].axvline(0, color=RED); ax[1].set_xlabel("Prediction error (% of actual price)"); ax[1].set_ylabel("Count"); ax[1].set_title("Distribution of percentage errors (median %.1f%%)" % np.median(np.abs(pe_)))
plt.tight_layout(); plt.savefig("figures/m3_pred_vs_actual.png"); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
ax[0].scatter(bestpred, resid, s=2, alpha=.2, color=BLUE); ax[0].axhline(0, color=RED); ax[0].set_xlabel("Predicted ln(price)"); ax[0].set_ylabel("Residual"); ax[0].set_title("Residuals vs predicted")
import statsmodels.api as sm
sm.qqplot(resid, line="s", ax=ax[1], markerfacecolor=BLUE, markeredgecolor=BLUE, alpha=.3, markersize=2); ax[1].set_title("Q-Q plot of test residuals")
xx = np.arange(10); ax[2].bar(xx - .2, dec.lin, .4, color=GREY, label="Linear"); ax[2].bar(xx + .2, dec.best, .4, color=GREEN, label="Gradient boosting"); ax[2].set_xticks(xx); ax[2].set_xticklabels(dec.index); ax[2].set_xlabel("Price decile (1 = cheapest)"); ax[2].set_ylabel("Median absolute % error"); ax[2].set_title("Error by price segment"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/m4_residuals.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); nm = {"log_carat": "carat (log)", "depth": "depth %", "table": "table %", "cut_ord": "cut", "color_ord": "colour", "clarity_ord": "clarity", "round_ct": "round weight", "just_below": "just below round"}
ax[0].barh([nm[i] for i in imp.index], imp.values, color=BLUE); ax[0].set_xlabel("Increase in test RMSE when shuffled"); ax[0].set_title("Permutation importance (gradient boosting)")
sns.barplot(data=ab, x="model", y="RMSE_log", hue="features", palette=[GREY, GREEN], ax=ax[1]); ax[1].set_ylabel("Test RMSE of ln(price)"); ax[1].set_xlabel(""); ax[1].set_title("Ablation: effect of round-weight features"); ax[1].legend(fontsize=8)
for c_ in ax[1].containers: ax[1].bar_label(c_, fmt="%.3f", fontsize=8, padding=2)
plt.tight_layout(); plt.savefig("figures/m5_importance_ablation.png"); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 3.8)); ax.plot(sizes, -trs.mean(axis=1), "o-", color=GREY, label="Training error"); ax.plot(sizes, -cvs_.mean(axis=1), "s-", color=GREEN, label="Cross-validation error")
ax.fill_between(sizes, -cvs_.mean(axis=1) - cvs_.std(axis=1), -cvs_.mean(axis=1) + cvs_.std(axis=1), color=GREEN, alpha=.2); ax.set_xlabel("Training set size"); ax.set_ylabel("RMSE of ln(price)"); ax.set_title("Learning curve (gradient boosting)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/m6_learning_curve.png"); plt.close()

# ===== CLASSIFICATION SUPPLEMENT =====
say("\n== CLASSIFICATION: high-value diamonds ==")
thr = tr.price.quantile(.75); ctr = (tr.price >= thr).astype(int); cte = (te.price >= thr).astype(int); say("threshold (75th percentile of train price): $%.0f | positive rate train %.1f%% test %.1f%%" % (thr, ctr.mean() * 100, cte.mean() * 100))
lg = Pipeline([("pre", lin_pre(NUM_LB)), ("m", LogisticRegression(max_iter=2000, C=1.0))]).fit(tr, ctr)
cs = RandomizedSearchCV(HistGradientBoostingClassifier(random_state=SEED), {"learning_rate": [0.05, 0.1, 0.2], "max_iter": [100, 200, 300], "max_leaf_nodes": [15, 31], "min_samples_leaf": [20, 50], "l2_regularization": [0, 1.0]}, n_iter=6, cv=3, scoring="roc_auc", random_state=SEED).fit(tr[Bf], ctr)
say("HGB classifier best", cs.best_params_, "CV AUC %.4f" % cs.best_score_)
rows = []; fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
for nm_, mdl, X_, col in [("Logistic regression", lg, te, GREY), ("Gradient boosting", cs.best_estimator_, te[Bf], GREEN)]:
    pr = mdl.predict(X_); pp = mdl.predict_proba(X_)[:, 1]; fpr, tpr, _ = roc_curve(cte, pp)
    rows.append(dict(model=nm_, accuracy=accuracy_score(cte, pr), precision=precision_score(cte, pr), recall=recall_score(cte, pr), F1=f1_score(cte, pr), ROC_AUC=roc_auc_score(cte, pp)))
    ax[1].plot(fpr, tpr, color=col, lw=2, label=f"{nm_} (AUC = {rows[-1]['ROC_AUC']:.3f})")
    if nm_ == "Gradient boosting": ConfusionMatrixDisplay(confusion_matrix(cte, pr), display_labels=["Standard", "High value"]).plot(ax=ax[0], cmap="Blues", colorbar=False); cm_gb = confusion_matrix(cte, pr)
ax[1].plot([0, 1], [0, 1], "k--", lw=.8); ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate"); ax[1].set_title("ROC curves"); ax[1].legend(fontsize=8, loc="lower right"); ax[0].set_title("Confusion matrix: gradient boosting")
plt.tight_layout(); plt.savefig("figures/m7_classification.png"); plt.close()
C = pd.DataFrame(rows).round(4); C.to_csv("classification_results.csv", index=False); say(C.to_string(index=False)); say("confusion matrix GB [[TN FP],[FN TP]]:", cm_gb.tolist())
say("\ntotal runtime %.0fs" % (time.time() - T0)); open("ml_log.txt", "w").write("\n".join(log))
