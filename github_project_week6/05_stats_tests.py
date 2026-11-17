"""Week 4 - Statistical Analysis and Hypothesis Testing (Diamonds, cleaned data)
Run:  python stats_tests.py     (needs diamonds_clean.csv from Week 2)
Outputs: stats_log.txt, results_summary.csv, figures/s*.png
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from scipy.stats import studentized_range
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
sns.set_style("whitegrid")
BLUE, GREEN, RED, ORANGE, GREY = "#2b6cb0", "#2f855a", "#c53030", "#dd6b20", "#718096"
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)
def fmtp(p): return "<0.001" if p < 0.001 else f"{p:.3f}"

CUT = ["Fair", "Good", "Very Good", "Premium", "Ideal"]; COLOR = list("DEFGHIJ")
CLARITY = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
df = pd.read_csv("diamonds_clean.csv")
df["cut"] = pd.Categorical(df.cut, CUT); df["color"] = pd.Categorical(df.color, COLOR); df["clarity"] = pd.Categorical(df.clarity, CLARITY)
n = len(df); rng = np.random.default_rng(42)
say("n =", n)
P = {}

# ===================== A. ASSUMPTION CHECKS =====================
say("\n== A. ASSUMPTIONS ==")
full = smf.ols("log_price ~ log_carat + C(cut) + C(color) + C(clarity)", df).fit()
res = full.resid
smp = df.sample(5000, random_state=1)
sw = stats.shapiro(smp.log_price); say("Shapiro-Wilk log_price (n=5000): W=%.3f p=%s" % (sw.statistic, fmtp(sw.pvalue)))
sw2 = stats.shapiro(res.sample(5000, random_state=1)); say("Shapiro-Wilk residuals (n=5000): W=%.4f p=%s" % (sw2.statistic, fmtp(sw2.pvalue)))
say("residual skew %.3f excess kurtosis %.3f" % (stats.skew(res), stats.kurtosis(res)))
dk = stats.normaltest(res); say("D'Agostino K2 residuals: %.1f p=%s" % (dk.statistic, fmtp(dk.pvalue)))
groups_cut = [g.log_price.values for _, g in df.groupby("cut", observed=True)]
lv = stats.levene(*groups_cut, center="median"); say("Levene (log_price by cut): W=%.1f p=%s" % (lv.statistic, fmtp(lv.pvalue)))
bp = het_breuschpagan(res, full.model.exog); say("Breusch-Pagan (full model): LM=%.1f p=%s" % (bp[0], fmtp(bp[1])))
say("group sds log_price by cut:", df.groupby("cut", observed=True).log_price.std().round(3).to_dict())
say("residual sd by fitted-value tercile:", pd.DataFrame({"t": pd.qcut(full.fittedvalues, 3, labels=False), "r": res}).groupby("t").r.std().round(3).to_dict())
say("min group n", int(df.cut.value_counts().min()), "| exact duplicate rows", int(df[["carat","cut","color","clarity","depth","table","price","x","y","z"]].duplicated().sum()))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
sm.qqplot(res, line="45", fit=True, ax=ax[0], markerfacecolor=BLUE, markeredgecolor=BLUE, alpha=.3, markersize=2); ax[0].set_title("Q-Q plot of regression residuals")
s2 = df.sample(8000, random_state=2); ax[1].scatter(full.fittedvalues[s2.index], res[s2.index], s=2, alpha=.25, color=BLUE); ax[1].axhline(0, color=RED); ax[1].set_title("Residuals vs fitted values"); ax[1].set_xlabel("Fitted ln(price)"); ax[1].set_ylabel("Residual")
sns.histplot(res, bins=60, ax=ax[2], color=GREEN, stat="density"); xs = np.linspace(res.min(), res.max(), 200); ax[2].plot(xs, stats.norm.pdf(xs, res.mean(), res.std()), color=RED); ax[2].set_title("Residual distribution vs normal curve"); ax[2].set_xlabel("Residual")
plt.tight_layout(); plt.savefig("figures/s1_assumptions.png"); plt.close()

# ===================== B. H1 =====================
say("\n== B. H1 ==")
def fisher_ci(r, n, z=1.96):
    zr = np.arctanh(r); se = 1 / np.sqrt(n - 3); return np.tanh(zr - z * se), np.tanh(zr + z * se)
pr_raw = stats.pearsonr(df.carat, df.price); pr_log = stats.pearsonr(df.log_carat, df.log_price); sp_raw = stats.spearmanr(df.carat, df.price)
rows = []
for name, r, p in [("Pearson, raw", pr_raw.statistic, pr_raw.pvalue), ("Spearman, raw", sp_raw.statistic, sp_raw.pvalue), ("Pearson, log-log", pr_log.statistic, pr_log.pvalue)]:
    lo, hi = fisher_ci(r, n); rows.append((name, r, lo, hi, p)); say("%-18s r=%.4f 95%% CI [%.4f, %.4f] p=%s" % (name, r, lo, hi, fmtp(p)))
P["H1 carat-price (Pearson log-log)"] = pr_log.pvalue
bs = []
for _ in range(300):
    i = rng.integers(0, n, n); bs.append(stats.spearmanr(df.carat.values[i], df.price.values[i]).statistic)
say("bootstrap Spearman 95%% CI [%.4f, %.4f]" % tuple(np.percentile(bs, [2.5, 97.5])))
m1 = smf.ols("log_price ~ log_carat", df).fit(cov_type="HC3"); b = m1.params["log_carat"]; ci = m1.conf_int().loc["log_carat"].values
say("log-log slope b=%.4f HC3 95%% CI [%.4f, %.4f] R2=%.4f t=%.1f" % (b, ci[0], ci[1], m1.rsquared, m1.tvalues["log_carat"]))
say("1%% heavier => price +%.2f%% (CI %.2f to %.2f)" % ((1.01 ** b - 1) * 100, (1.01 ** ci[0] - 1) * 100, (1.01 ** ci[1] - 1) * 100))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
ax[0].hexbin(df.log_carat, df.log_price, gridsize=60, bins="log", cmap="viridis", mincnt=1); xs = np.linspace(df.log_carat.min(), df.log_carat.max(), 50)
ax[0].plot(xs, m1.params["Intercept"] + b * xs, color=RED, lw=2, label=f"fit: slope = {b:.2f}"); ax[0].legend(); ax[0].set_xlabel("ln(carat)"); ax[0].set_ylabel("ln(price)"); ax[0].set_title("H1: log-log regression line (R\u00b2 = %.3f)" % m1.rsquared)
for i, (name, r, lo, hi, p) in enumerate(rows):
    ax[1].errorbar(r, i, xerr=[[r - lo], [hi - r]], fmt="o", color=BLUE, capsize=4); ax[1].text(r, i + .25, f"{r:.3f}", ha="center", fontsize=8)
ax[1].set_yticks(range(3)); ax[1].set_yticklabels([r[0] for r in rows]); ax[1].set_xlim(0.9, 1.0); ax[1].set_xlabel("Correlation coefficient with 95% CI (Fisher z)"); ax[1].set_title("H1: correlation estimates"); ax[1].invert_yaxis()
plt.tight_layout(); plt.savefig("figures/s2_h1_carat.png"); plt.close()

# ===================== C. H2 =====================
say("\n== C. H2 (cut) ==")
aw = anova_oneway(df.log_price.values, df.cut.astype(str).values, use_var="unequal", welch_correction=True)
say("Welch ANOVA log_price ~ cut: F=%.1f df=(%.0f, %.0f) p=%s" % (aw.statistic, aw.df_num, aw.df_denom, fmtp(aw.pvalue)))
kw = stats.kruskal(*groups_cut); say("Kruskal-Wallis: H=%.1f p=%s epsilon2=%.4f" % (kw.statistic, fmtp(kw.pvalue), kw.statistic / (n - 1)))
ss_b = sum(len(g) * (g.mean() - df.log_price.mean()) ** 2 for g in groups_cut); ss_t = ((df.log_price - df.log_price.mean()) ** 2).sum()
say("eta2 (unadjusted, cut on ln price) = %.4f" % (ss_b / ss_t)); P["H2a cut unadjusted (Welch ANOVA)"] = aw.pvalue
means = df.groupby("cut", observed=True).log_price.agg(["mean", "var", "count"]); gh = []
for i, a_ in enumerate(CUT):
    for j, b_ in enumerate(CUT):
        if j <= i: continue
        ma, mb = means.loc[a_], means.loc[b_]; se2 = ma["var"] / ma["count"] + mb["var"] / mb["count"]
        t = (mb["mean"] - ma["mean"]) / np.sqrt(se2); dfw = se2 ** 2 / ((ma["var"] / ma["count"]) ** 2 / (ma["count"] - 1) + (mb["var"] / mb["count"]) ** 2 / (mb["count"] - 1))
        gh.append((a_, b_, mb["mean"] - ma["mean"], studentized_range.sf(abs(t) * np.sqrt(2), 5, dfw)))
ghd = pd.DataFrame(gh, columns=["g1", "g2", "diff", "p"]); say("Games-Howell pairs (diff = mean ln price g2 - g1):\n" + ghd.round(3).to_string(index=False)); say("pairs significant at 0.05:", int((ghd.p < .05).sum()), "of", len(ghd))
a = df[df.cut == "Ideal"].log_price; f_ = df[df.cut == "Fair"].log_price
tt = stats.ttest_ind(a, f_, equal_var=False); sp = np.sqrt(((len(a) - 1) * a.var() + (len(f_) - 1) * f_.var()) / (len(a) + len(f_) - 2)); d = (a.mean() - f_.mean()) / sp
dm = a.mean() - f_.mean(); se = np.sqrt(a.var() / len(a) + f_.var() / len(f_)); dfw = tt.df
say("Welch t Ideal vs Fair (ln price): t=%.1f df=%.0f p=%s diff=%.3f CI [%.3f, %.3f] Cohen d=%.2f" % (tt.statistic, dfw, fmtp(tt.pvalue), dm, dm - stats.t.ppf(.975, dfw) * se, dm + stats.t.ppf(.975, dfw) * se, d))
say("  as price ratio: %.1f%% (CI %.1f%% to %.1f%%)" % ((np.exp(dm) - 1) * 100, (np.exp(dm - 1.96 * se) - 1) * 100, (np.exp(dm + 1.96 * se) - 1) * 100))
cm = smf.ols("log_price ~ log_carat + C(cut)", df).fit(cov_type="HC3"); cmc = smf.ols("log_price ~ log_carat + C(cut)", df).fit()
at = anova_lm(cmc, typ=2); ssc = at.loc["C(cut)", "sum_sq"]; sse = at.loc["Residual", "sum_sq"]
say("H2b ANCOVA cut | log_carat: F=%.1f p=%s partial eta2=%.4f" % (at.loc["C(cut)", "F"], fmtp(at.loc["C(cut)", "PR(>F)"]), ssc / (ssc + sse))); P["H2b cut adjusted for carat (ANCOVA)"] = at.loc["C(cut)", "PR(>F)"]
ci2 = cm.conf_int(); adj = []
for lv_ in CUT[1:]:
    k = f"C(cut)[T.{lv_}]"; adj.append((lv_, (np.exp(cm.params[k]) - 1) * 100, (np.exp(ci2.loc[k, 0]) - 1) * 100, (np.exp(ci2.loc[k, 1]) - 1) * 100, cm.pvalues[k]))
say("adjusted cut effects vs Fair (%, HC3 CI):\n" + pd.DataFrame(adj, columns=["cut", "pct", "lo", "hi", "p"]).round(2).to_string(index=False))
cm0 = smf.ols("log_price ~ C(cut)", df).fit(cov_type="HC3"); ci0 = cm0.conf_int(); un = []
for lv_ in CUT[1:]:
    k = f"C(cut)[T.{lv_}]"; un.append((lv_, (np.exp(cm0.params[k]) - 1) * 100, (np.exp(ci0.loc[k, 0]) - 1) * 100, (np.exp(ci0.loc[k, 1]) - 1) * 100))
say("unadjusted cut effects vs Fair (%):", [(u[0], round(u[1], 1), round(u[2], 1), round(u[3], 1)) for u in un])
X = pd.get_dummies(df.cut, drop_first=True, dtype=float); X.insert(0, "lc", df.log_carat.values); X.insert(0, "c", 1.0); Xv = X.values; yv = df.log_price.values; bsd = []
for _ in range(300):
    i = rng.integers(0, n, n); beta = np.linalg.lstsq(Xv[i], yv[i], rcond=None)[0]; bsd.append((np.exp(beta[-1]) - 1) * 100)
say("bootstrap 95%% CI Ideal vs Fair adjusted: [%.1f%%, %.1f%%]" % tuple(np.percentile(bsd, [2.5, 97.5])))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9)); yy = np.arange(4)
ax[0].errorbar([u[1] for u in un], yy + .12, xerr=[[u[1] - u[2] for u in un], [u[3] - u[1] for u in un]], fmt="o", color=RED, capsize=3, label="Unadjusted")
ax[0].errorbar([u[1] for u in adj], yy - .12, xerr=[[u[1] - u[2] for u in adj], [u[3] - u[1] for u in adj]], fmt="s", color=GREEN, capsize=3, label="Adjusted for carat")
ax[0].axvline(0, color="k", lw=.8); ax[0].set_yticks(yy); ax[0].set_yticklabels(CUT[1:]); ax[0].set_xlabel("Price difference vs Fair cut (%), 95% CI"); ax[0].set_title("H2: cut effects, unadjusted vs carat-adjusted"); ax[0].legend(fontsize=8); ax[0].invert_yaxis()
pos = np.arange(5); ax[1].bar(pos, [means.loc[c, "mean"] for c in CUT], yerr=[1.96 * np.sqrt(means.loc[c, "var"] / means.loc[c, "count"]) for c in CUT], color=BLUE, capsize=4); ax[1].set_xticks(pos); ax[1].set_xticklabels(CUT); ax[1].set_ylim(7, 8.2); ax[1].set_ylabel("Mean ln(price)"); ax[1].set_title("H2a: mean ln(price) by cut, 95% CI")
plt.tight_layout(); plt.savefig("figures/s3_h2_cut.png"); plt.close()

# ===================== D. H3 =====================
say("\n== D. H3 ==")
fa = anova_lm(full, typ=2); sse = fa.loc["Residual", "sum_sq"]; pe = {}
for term, lab in [("log_carat", "carat"), ("C(cut)", "cut"), ("C(color)", "colour"), ("C(clarity)", "clarity")]:
    ss = fa.loc[term, "sum_sq"]; pe[lab] = ss / (ss + sse); say("%-8s F=%.1f df=%d p=%s partial eta2=%.4f" % (lab, fa.loc[term, "F"], fa.loc[term, "df"], fmtp(fa.loc[term, "PR(>F)"]), pe[lab]))
P["H3 colour (ANCOVA)"] = fa.loc["C(color)", "PR(>F)"]; P["H3 clarity (ANCOVA)"] = fa.loc["C(clarity)", "PR(>F)"]
df["res0"] = smf.ols("log_price ~ log_carat", df).fit().resid
df["color_ord"] = df.color.map({k: 6 - i for i, k in enumerate(COLOR)}).astype(float); df["clarity_ord"] = df.clarity.map({k: i for i, k in enumerate(CLARITY)}).astype(float); df["cut_ord"] = df.cut.map({k: i for i, k in enumerate(CUT)}).astype(float)
for v in ["color_ord", "clarity_ord", "cut_ord"]:
    r = stats.spearmanr(df[v], df.res0); lo, hi = fisher_ci(r.statistic, n); say("Spearman trend %s vs carat-adjusted ln price: rho=%.3f CI [%.3f, %.3f] p=%s" % (v, r.statistic, lo, hi, fmtp(r.pvalue)))
cr = full.get_robustcov_results("HC3"); ci3 = pd.DataFrame(cr.conf_int(), index=full.params.index)
def eff(term): return (np.exp(full.params[term]) - 1) * 100, (np.exp(ci3.loc[term, 0]) - 1) * 100, (np.exp(ci3.loc[term, 1]) - 1) * 100
say("J vs D colour: %.1f%% [%.1f, %.1f] | IF vs I1 clarity: %.1f%% [%.1f, %.1f] | Ideal vs Fair: %.1f%% [%.1f, %.1f]" % (*eff("C(color)[T.J]"), *eff("C(clarity)[T.IF]"), *eff("C(cut)[T.Ideal]")))
say("Cohen f2 (full vs carat only):", round((full.rsquared - m1.rsquared) / (1 - full.rsquared), 3), "| R2 full %.4f carat-only %.4f" % (full.rsquared, m1.rsquared))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); lab = list(pe.keys())
ax[0].barh(lab[::-1], [pe[k] for k in lab][::-1], color=BLUE)
for i, k in enumerate(lab[::-1]): ax[0].text(pe[k] + .01, i, f"{pe[k]:.3f}", va="center", fontsize=8)
ax[0].set_xlim(0, 1.1); ax[0].set_xlabel("Partial eta squared"); ax[0].set_title("H2/H3: effect size of each factor")
for v, nm, col, order in [("color", "Colour (D best to J worst)", ORANGE, COLOR), ("clarity", "Clarity (I1 worst to IF best)", GREEN, CLARITY)]:
    g = df.groupby(v, observed=True).res0.agg(["mean", "sem"]).reindex(order)
    ax[1].errorbar(range(len(order)), (np.exp(g["mean"]) - 1) * 100, yerr=1.96 * g["sem"] * np.exp(g["mean"]) * 100, marker="o", color=col, capsize=3, label=nm)
ax[1].axhline(0, color="k", lw=.8); ax[1].set_xlabel("Position in grade list (left = D or I1)"); ax[1].set_ylabel("Carat-adjusted price premium (%)"); ax[1].set_title("H3: monotonic trend across grades"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/s4_h3_grades.png"); plt.close()

# ===================== E. H4 =====================
say("\n== E. H4 ==")
d4 = df.copy(); d4["depth_z"] = (d4.depth - d4.depth.mean()) / d4.depth.std(); d4["table_z"] = (d4.table - d4.table.mean()) / d4.table.std()
m4 = smf.ols("log_price ~ log_carat + C(cut) + C(color) + C(clarity) + depth_z + table_z", d4).fit(cov_type="HC3"); delta = np.log(1.02); h4 = []
for v in ["depth_z", "table_z"]:
    bb, se = m4.params[v], m4.bse[v]; lo, hi = bb - 1.645 * se, bb + 1.645 * se
    ptost = max(1 - stats.norm.cdf((bb + delta) / se), stats.norm.cdf((bb - delta) / se)); ci95 = m4.conf_int().loc[v].values
    h4.append((v, (np.exp(bb) - 1) * 100, (np.exp(ci95[0]) - 1) * 100, (np.exp(ci95[1]) - 1) * 100, m4.pvalues[v], ptost, (np.exp(lo) - 1) * 100, (np.exp(hi) - 1) * 100))
    say("%s: effect per 1 SD = %.2f%% 95%% CI [%.2f, %.2f] p(diff from 0)=%s | TOST (margin +/-2%%) p=%s | 90%% CI [%.2f, %.2f]" % (v, *h4[-1][1:4], fmtp(m4.pvalues[v]), fmtp(ptost), h4[-1][6], h4[-1][7]))
base = smf.ols("log_price ~ log_carat + C(cut) + C(color) + C(clarity)", d4).fit()
say("R2 without depth/table %.5f with %.5f, delta %.5f; partial R2 of the pair %.5f" % (base.rsquared, m4.rsquared, m4.rsquared - base.rsquared, (m4.rsquared - base.rsquared) / (1 - base.rsquared)))
say("SD depth %.2f SD table %.2f" % (df.depth.std(), df.table.std())); P["H4 depth (t-test, differs from 0)"] = m4.pvalues["depth_z"]; P["H4 table (t-test, differs from 0)"] = m4.pvalues["table_z"]
fig, ax = plt.subplots(figsize=(7.5, 3.4)); ax.axvspan(-2, 2, color="#c6f6d5", alpha=.6, label="Equivalence zone (\u00b12%)")
for i, r in enumerate(h4):
    ax.errorbar(r[1], i, xerr=[[r[1] - r[2]], [r[3] - r[1]]], fmt="o", color=BLUE, capsize=4); ax.text(r[1], i + .22, f"{r[1]:.2f}%", ha="center", fontsize=8)
ax.axvline(0, color="k", lw=.8); ax.set_yticks([0, 1]); ax.set_yticklabels(["depth % (per 1 SD)", "table % (per 1 SD)"]); ax.set_xlim(-3, 3); ax.set_xlabel("Price change per 1 SD, 95% CI (%)"); ax.set_title("H4: depth and table effects vs the \u00b12% equivalence zone"); ax.legend(loc="lower right", fontsize=8); ax.invert_yaxis()
plt.tight_layout(); plt.savefig("figures/s5_h4_equivalence.png"); plt.close()

# ===================== F. H5 =====================
say("\n== F. H5 ==")
m0f = smf.ols("log_price ~ log_carat", df).fit(); cmp_ = anova_lm(m0f, full)
say("Nested F-test carat-only vs full: F=%.1f p=%s | R2 %.4f -> %.4f | AIC %.0f -> %.0f" % (cmp_.F.iloc[1], fmtp(cmp_["Pr(>F)"].iloc[1]), m0f.rsquared, full.rsquared, m0f.aic, full.aic)); P["H5 full vs carat-only (nested F)"] = cmp_["Pr(>F)"].iloc[1]
D = pd.get_dummies(df[["cut", "color", "clarity"]], drop_first=True, dtype=float); X1 = pd.concat([df[["log_carat"]], D], axis=1).values; X0 = df[["log_carat"]].values; y = df.log_price.values
kf = KFold(10, shuffle=True, random_state=42); r0, r1 = [], []
for tr, te in kf.split(X1):
    for X_, out in [(X0, r0), (X1, r1)]:
        m = LinearRegression().fit(X_[tr], y[tr]); out.append(np.sqrt(np.mean((y[te] - m.predict(X_[te])) ** 2)))
d_ = np.array(r0) - np.array(r1); k = 10; tcv = d_.mean() / np.sqrt((1 / k + 1 / (k - 1)) * d_.var(ddof=1)); pcv = 2 * stats.t.sf(abs(tcv), k - 1); tpl = stats.ttest_rel(r0, r1)
say("10-fold CV RMSE(ln price): carat-only %.4f full %.4f | mean paired diff %.4f | naive paired t=%.1f p=%s | Nadeau-Bengio corrected t=%.1f p=%s" % (np.mean(r0), np.mean(r1), d_.mean(), tpl.statistic, fmtp(tpl.pvalue), tcv, fmtp(pcv)))
say("RMSE reduction %.1f%%; Cohen d of paired diffs %.1f" % ((1 - np.mean(r1) / np.mean(r0)) * 100, d_.mean() / d_.std(ddof=1)))

# ===================== G. H6 =====================
say("\n== G. H6 round weights (exploratory) ==")
ROUND = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
df["round_ct"] = df.carat.round(2).isin(ROUND).astype(int); say("share round weights %.1f%% (n=%d)" % (df.round_ct.mean() * 100, df.round_ct.sum()))
rr = full.resid[df.round_ct == 1]; nr = full.resid[df.round_ct == 0]
tw = stats.ttest_ind(rr, nr, equal_var=False); dmr = rr.mean() - nr.mean(); ser = np.sqrt(rr.var() / len(rr) + nr.var() / len(nr)); dd = dmr / np.sqrt((rr.var() + nr.var()) / 2)
say("Welch t residuals round vs non-round: t=%.1f p=%s diff=%.4f CI [%.4f, %.4f] Cohen d=%.2f" % (tw.statistic, fmtp(tw.pvalue), dmr, dmr - 1.96 * ser, dmr + 1.96 * ser, dd))
mw = stats.mannwhitneyu(rr, nr); say("Mann-Whitney U p=%s | rank-biserial r=%.3f" % (fmtp(mw.pvalue), 1 - 2 * mw.statistic / (len(rr) * len(nr))))
rm = smf.ols("log_price ~ log_carat + C(cut) + C(color) + C(clarity) + round_ct", df).fit(cov_type="HC3"); ci6 = rm.conf_int().loc["round_ct"].values
say("regression indicator: +%.2f%% CI [%.2f, %.2f] p=%s" % ((np.exp(rm.params["round_ct"]) - 1) * 100, (np.exp(ci6[0]) - 1) * 100, (np.exp(ci6[1]) - 1) * 100, fmtp(rm.pvalues["round_ct"]))); P["H6 round weights (Welch t)"] = tw.pvalue
lab_ = df.round_ct.values.astype(bool); rv = full.resid.values; obs = rv[lab_].mean() - rv[~lab_].mean(); perm = np.empty(5000)
for i in range(5000):
    pl_ = rng.permutation(lab_); perm[i] = rv[pl_].mean() - rv[~pl_].mean()
pp = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (len(perm) + 1); say("permutation test (5000): observed diff %.4f, null sd %.4f, p=%.4f (min possible %.4f)" % (obs, perm.std(), pp, 1 / 5001))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); fx = np.arange(10)
ax[0].plot(fx, r0, "o-", color=RED, label="Carat only"); ax[0].plot(fx, r1, "s-", color=GREEN, label="Carat + cut + colour + clarity")
ax[0].set_xticks(fx); ax[0].set_xticklabels([str(i + 1) for i in fx]); ax[0].set_xlabel("Cross-validation fold"); ax[0].set_ylabel("RMSE of ln(price)"); ax[0].set_title("H5: 10-fold CV error by fold"); ax[0].legend(fontsize=8)
sns.histplot(perm, bins=40, ax=ax[1], color=GREY); ax[1].axvline(obs, color=RED, lw=2, label=f"observed = {obs:.3f}"); ax[1].set_xlabel("Difference in mean residual (round - other)"); ax[1].set_title("H6: permutation null distribution"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/s6_h5_h6.png"); plt.close()

# ===================== H. H7 =====================
say("\n== H. H7 chi-square ==")
df["size_class"] = pd.Categorical(df.size_class, ["small", "medium", "large", "very_large", "exceptional"]); chi_out = {}
for v in ["cut", "color", "clarity"]:
    ct = pd.crosstab(df[v], df.size_class); chi2, p, dof, exp = stats.chi2_contingency(ct); V = np.sqrt(chi2 / (n * (min(ct.shape) - 1))); chi_out[v] = (chi2, p, dof, V, exp.min())
    say("%s x size_class: chi2=%.1f dof=%d p=%s Cramer V=%.3f min expected=%.1f" % (v, chi2, dof, fmtp(p), V, exp.min()))
P["H7 cut x size (chi-square)"] = chi_out["cut"][1]
ct = pd.crosstab(df.cut, df.size_class); chi2, p, dof, exp = stats.chi2_contingency(ct); sr = (ct - exp) / np.sqrt(exp)
say("share small by cut %:", (ct["small"] / ct.sum(axis=1) * 100).round(1).to_dict()); say("share large+ by cut %:", ((ct[["large", "very_large", "exceptional"]].sum(axis=1)) / ct.sum(axis=1) * 100).round(1).to_dict())
say("size_class counts:", df.size_class.value_counts().reindex(["small", "medium", "large", "very_large", "exceptional"]).to_dict())
fig, ax = plt.subplots(figsize=(7.2, 3.9)); sns.heatmap(sr, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "Standardised residual"})
ax.set_title("H7: cut vs size class (chi-square = %.0f, Cramer's V = %.2f)" % (chi_out["cut"][0], chi_out["cut"][3])); ax.set_xlabel("Size class (by carat)"); ax.set_ylabel("Cut")
plt.tight_layout(); plt.savefig("figures/s7_h7_chi2.png"); plt.close()

# ===================== I. MULTIPLE TESTING =====================
say("\n== I. MULTIPLE TESTING (Holm) ==")
names = list(P.keys()); pv = np.array([max(P[k], 1e-300) for k in names]); rej, padj, _, _ = multipletests(pv, method="holm")
mt = pd.DataFrame({"test": names, "raw_p": [("<1e-300" if x <= 1e-300 else f"{x:.3g}") for x in pv], "holm_p": [f"{x:.3g}" for x in padj], "reject_H0": rej}); say(mt.to_string(index=False))
mt.to_csv("results_summary.csv", index=False); open("stats_log.txt", "w").write("\n".join(log))
