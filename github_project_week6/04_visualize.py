"""Week 3 - Data Visualization and Reporting (Diamonds, cleaned data from Week 2)
Run:  python visualize.py      (needs diamonds_clean.csv from Week 2)
Outputs: figures/*.png, viz_log.txt, interactive_explorer.html
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import r2_score, mean_absolute_error

plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
sns.set_style("whitegrid")
BLUE, GREEN, RED, GREY = "#2b6cb0", "#2f855a", "#c53030", "#718096"
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

CUT = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR = list("DEFGHIJ")
CLARITY = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
df = pd.read_csv("diamonds_clean.csv")
df["cut"] = pd.Categorical(df.cut, CUT); df["color"] = pd.Categorical(df.color, COLOR); df["clarity"] = pd.Categorical(df.clarity, CLARITY)
say("rows", len(df))

# ---------- Fig 1  BAR ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
cnt = df.cut.value_counts().reindex(CUT)
b = ax[0].bar(CUT, cnt.values, color=BLUE); ax[0].bar_label(b, fmt="{:,.0f}", fontsize=8, padding=2)
ax[0].set_title("Number of diamonds by cut"); ax[0].set_xlabel("Cut (worst to best)"); ax[0].set_ylabel("Count")
med = df.groupby("cut", observed=True).price.median().reindex(CUT)
b = ax[1].bar(CUT, med.values, color=GREEN); ax[1].bar_label(b, fmt="${:,.0f}", fontsize=8, padding=2)
ax[1].set_title("Median price by cut"); ax[1].set_xlabel("Cut (worst to best)"); ax[1].set_ylabel("Median price (USD)")
plt.tight_layout(); plt.savefig("figures/v1_bar_cut.png"); plt.close()
say("count by cut", cnt.to_dict()); say("median price by cut", med.to_dict())
say("share premium+ideal %", round((cnt["Premium"] + cnt["Ideal"]) / cnt.sum() * 100, 1))

# ---------- Fig 2  DISTRIBUTIONS ----------
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for a, col, t, xl, colr in zip(ax, ["price", "log_price", "carat"], ["Price", "Log of price", "Carat weight"], ["Price (USD)", "ln(price)", "Carat"], [BLUE, GREEN, "#805ad5"]):
    sns.histplot(df[col], bins=50, ax=a, color=colr); m = df[col].median()
    a.axvline(m, color=RED, ls="--", lw=1.2, label=f"median = {m:,.2f}"); a.set_title(t); a.set_xlabel(xl); a.set_ylabel("Count"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/v2_distributions.png"); plt.close()
say("skew price", round(df.price.skew(), 2), "log_price", round(df.log_price.skew(), 2), "carat", round(df.carat.skew(), 2), "| mean price", round(df.price.mean()), "median", round(df.price.median()))
say("share of diamonds <= 1 carat %", round((df.carat <= 1).mean() * 100, 1), "| price < 5000 %", round((df.price < 5000).mean() * 100, 1))

# ---------- Fig 3  LINE ----------
edges = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
df["cbin"] = pd.cut(df.carat, edges)
mid = {iv: (iv.left + iv.right) / 2 for iv in df.cbin.cat.categories}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
pal = sns.color_palette("viridis", 8)
for i, cl in enumerate(CLARITY):
    g = df[df.clarity == cl].groupby("cbin", observed=True).price.agg(["median", "count"]); g = g[g["count"] >= 15]
    ax[0].plot([mid[k] for k in g.index], g["median"], marker="o", ms=4, color=pal[i], label=cl)
ax[0].set_yscale("log"); ax[0].set_title("Median price by carat band and clarity"); ax[0].set_xlabel("Carat band midpoint"); ax[0].set_ylabel("Median price (USD, log scale)"); ax[0].legend(title="Clarity", fontsize=7, ncol=2)
pal2 = sns.color_palette("plasma", 5)
for i, cu in enumerate(CUT):
    g = df[df.cut == cu].groupby("cbin", observed=True).price.agg(["median", "count"]); g = g[g["count"] >= 15]
    ax[1].plot([mid[k] for k in g.index], g["median"], marker="o", ms=4, color=pal2[i], label=cu)
ax[1].set_yscale("log"); ax[1].set_title("Median price by carat band and cut"); ax[1].set_xlabel("Carat band midpoint"); ax[1].set_ylabel("Median price (USD, log scale)"); ax[1].legend(title="Cut", fontsize=7)
plt.tight_layout(); plt.savefig("figures/v3_line_trend.png"); plt.close()
t = df[(df.cbin == pd.Interval(0.8, 1.0))].groupby("clarity", observed=True).price.median().round(0)
say("median price 0.8-1.0 ct by clarity", t.to_dict())
t = df[(df.cbin == pd.Interval(0.8, 1.0))].groupby("cut", observed=True).price.median().round(0)
say("median price 0.8-1.0 ct by cut", t.to_dict())
tt = df.groupby("cbin", observed=True).price.median().round(0); say("median price by carat band", {str(k): v for k, v in tt.items()})

# ---------- Fig 4  SCATTER ----------
s = df.sample(8000, random_state=3)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sns.scatterplot(data=s, x="carat", y="price", hue="clarity", palette="viridis", s=7, alpha=.5, linewidth=0, ax=ax[0], legend=False)
ax[0].set_title("Raw scale: curved, fan-shaped relation"); ax[0].set_xlabel("Carat"); ax[0].set_ylabel("Price (USD)")
sns.scatterplot(data=s, x="log_carat", y="log_price", hue="clarity", palette="viridis", s=7, alpha=.5, linewidth=0, ax=ax[1])
ax[1].set_title("Log-log scale: near-linear, even spread"); ax[1].set_xlabel("ln(carat)"); ax[1].set_ylabel("ln(price)")
ax[1].legend(title="Clarity", fontsize=7, ncol=2, loc="lower right", markerscale=1.5)
plt.tight_layout(); plt.savefig("figures/v4_scatter.png"); plt.close()
say("corr carat-price", round(df.carat.corr(df.price), 3), "| log-log", round(df.log_carat.corr(df.log_price), 3))
sl = np.polyfit(df.log_carat, df.log_price, 1); say("log-log slope (elasticity)", round(sl[0], 3), "intercept", round(sl[1], 3))

# ---------- Fig 5  HEATMAP (correlation) ----------
cols = ["carat", "depth", "table", "x", "y", "z", "volume", "cut_ord", "color_ord", "clarity_ord", "price"]
cm = df[cols].corr()
fig, ax = plt.subplots(figsize=(8, 6.2))
sns.heatmap(cm, mask=np.triu(np.ones_like(cm, dtype=bool), 1), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, linewidths=.5, cbar_kws={"label": "Pearson r", "shrink": .8}, ax=ax, annot_kws={"size": 8})
ax.set_title("Correlation heatmap (cleaned data; ordinal grades: higher = better)")
plt.tight_layout(); plt.savefig("figures/v5_heatmap_corr.png"); plt.close()
say("corr price:", cm["price"].round(2).to_dict()); say("corr carat with cut_ord/color_ord/clarity_ord:", cm.loc["carat", ["cut_ord", "color_ord", "clarity_ord"]].round(2).to_dict())

# ---------- Fig 6  HEATMAPS (raw vs carat-adjusted) ----------
df["band"] = pd.cut(df.carat, [0, .5, 1, 1.5, 6], labels=["<=0.5", "0.5-1", "1-1.5", ">1.5"])
m0 = smf.ols("log_price ~ log_carat", df).fit(); df["res0"] = m0.resid
p1 = df.pivot_table(index="band", columns="clarity", values="price", aggfunc="median", observed=True)
p2 = df.pivot_table(index="clarity", columns="color", values="res0", aggfunc="mean", observed=True); p2 = (np.exp(p2) - 1) * 100
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
sns.heatmap(p1, annot=True, fmt=",.0f", cmap="YlGnBu", ax=ax[0], cbar_kws={"label": "Median price (USD)"}, annot_kws={"size": 8}); ax[0].set_title("Raw: median price by carat band and clarity"); ax[0].set_xlabel("Clarity (worst to best)"); ax[0].set_ylabel("Carat band")
sns.heatmap(p2, annot=True, fmt=".0f", cmap="RdBu", center=0, ax=ax[1], cbar_kws={"label": "% vs average diamond of same carat"}, annot_kws={"size": 8}); ax[1].set_title("Carat-adjusted premium (%) by clarity and colour"); ax[1].set_xlabel("Colour (D best to J worst)"); ax[1].set_ylabel("Clarity")
plt.tight_layout(); plt.savefig("figures/v6_heatmap_adjusted.png"); plt.close()
say("raw median by band x clarity:\n", p1.round(0).to_string()); say("adjusted premium % clarity x colour:\n", p2.round(0).to_string())

# ---------- Fig 7  BAR (confounding) ----------
a0 = smf.ols("log_price ~ C(cut)", df).fit(); a1 = smf.ols("log_price ~ log_carat + C(cut)", df).fit()
e0 = (np.exp(a0.params.filter(like="cut")) - 1) * 100; e1 = (np.exp(a1.params.filter(like="cut")) - 1) * 100
w = pd.DataFrame({"cut": CUT[1:], "Ignoring carat": e0.values, "Controlling for carat": e1.values}).melt("cut", var_name="Model", value_name="pct")
fig, ax = plt.subplots(figsize=(7.5, 4))
sns.barplot(data=w, x="cut", y="pct", hue="Model", palette=[RED, GREEN], ax=ax)
for c in ax.containers: ax.bar_label(c, fmt="%.0f%%", fontsize=8, padding=2)
ax.axhline(0, color="k", lw=.8); ax.set_title("Cut effect on price relative to Fair cut"); ax.set_xlabel("Cut"); ax.set_ylabel("Price difference vs Fair (%)"); ax.legend(fontsize=8, loc="upper left"); ax.set_ylim(-50, 52)
plt.tight_layout(); plt.savefig("figures/v7_bar_confounding.png"); plt.close()
say("cut effect ignoring carat %", e0.round(1).values.tolist(), "| controlling %", e1.round(1).values.tolist())

# ---------- Fig 8  FOREST PLOT ----------
f = smf.ols("log_price ~ log_carat + C(cut) + C(color) + C(clarity)", df).fit()
ci = f.conf_int(); rows = []
for term in f.params.index:
    if term.startswith("C("):
        grp = term.split("(")[1].split(")")[0]; lvl = term.split("[T.")[1].rstrip("]")
        rows.append((grp, lvl, (np.exp(f.params[term]) - 1) * 100, (np.exp(ci.loc[term, 0]) - 1) * 100, (np.exp(ci.loc[term, 1]) - 1) * 100))
fr = pd.DataFrame(rows, columns=["group", "level", "pct", "lo", "hi"])
fig, ax = plt.subplots(figsize=(7.5, 6.2)); y = 0; yt, yl = [], []
colors = {"cut": BLUE, "color": "#dd6b20", "clarity": GREEN}
for grp in ["cut", "color", "clarity"]:
    for _, r in fr[fr.group == grp].iterrows():
        ax.errorbar(r.pct, y, xerr=[[r.pct - r.lo], [r.hi - r.pct]], fmt="o", color=colors[grp], capsize=3, ms=5); yt.append(y); yl.append(f"{grp}: {r.level}"); y += 1
    y += 0.6
ax.axvline(0, color="k", lw=.8); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("Price difference at the same carat (%), 95% CI"); ax.set_title("Quality grade effects after controlling for carat\n(reference: Fair cut, D colour, I1 clarity)")
plt.tight_layout(); plt.savefig("figures/v8_forest.png"); plt.close()
say("forest table:\n", fr.round(1).to_string(index=False)); say("model R2", round(f.rsquared, 4))

# ---------- Fig 9  BOXPLOTS within carat bands ----------
fig, ax = plt.subplots(1, 4, figsize=(12, 3.8))
for a, bnd in zip(ax, ["<=0.5", "0.5-1", "1-1.5", ">1.5"]):
    d = df[df.band == bnd]
    sns.boxplot(data=d, x="cut", y="price", order=CUT, ax=a, color="#90cdf4", showfliers=False, linewidth=.8)
    a.set_title(f"{bnd} carat (n={len(d):,})"); a.set_xlabel(""); a.set_ylabel("Price (USD)" if bnd == "<=0.5" else ""); a.tick_params(axis="x", rotation=40, labelsize=7)
plt.tight_layout(); plt.savefig("figures/v9_box_bands.png"); plt.close()
say("median price band x cut:\n", df.pivot_table(index="band", columns="cut", values="price", aggfunc="median", observed=True).round(0).to_string())

# ---------- Fig 10  NON-LINEARITY residuals ----------
lin = smf.ols("log_price ~ log_carat + C(cut) + C(color) + C(clarity)", df).fit()
quad = smf.ols("log_price ~ log_carat + I(log_carat**2) + C(cut) + C(color) + C(clarity)", df).fit()
df["r_lin"] = lin.resid; df["r_quad"] = quad.resid; df["lbin"] = pd.qcut(df.log_carat, 25, duplicates="drop")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
mx = {}
for a, col, t in zip(ax, ["r_lin", "r_quad"], ["Linear in ln(carat)", "With ln(carat) squared"]):
    g = df.groupby("lbin", observed=True).agg(x=("log_carat", "mean"), m=(col, "mean"), se=(col, "sem"))
    a.errorbar(g.x, g.m, yerr=1.96 * g.se, fmt="o-", color=BLUE, ms=4, capsize=2); a.axhline(0, color=RED, ls="--"); a.set_title("Mean residual by carat bin: " + t); a.set_xlabel("ln(carat)"); a.set_ylabel("Mean residual of ln(price)")
    mx[col] = round(g.m.abs().max(), 3)
plt.tight_layout(); plt.savefig("figures/v10_nonlinearity.png"); plt.close()
say("max |mean residual| linear", mx["r_lin"], "quadratic", mx["r_quad"], "| R2", round(lin.rsquared, 4), round(quad.rsquared, 4), "| AIC", round(lin.aic), round(quad.aic))
g = df.groupby("lbin", observed=True).agg(x=("log_carat", "mean"), m=("r_lin", "mean")); say("lin residual first/last bins", g.m.round(3).iloc[[0, 1, 2, -3, -2, -1]].to_dict())

# ---------- Fig 11  MODEL INTERPRETATION ----------
feats = ["log_carat", "depth", "table", "cut_ord", "color_ord", "clarity_ord"]
df[feats] = df[feats].astype(float)
Xtr, Xte, ytr, yte = train_test_split(df[feats], df.log_price, test_size=.2, random_state=42)
gb = HistGradientBoostingRegressor(random_state=42).fit(Xtr, ytr); lr = LinearRegression().fit(Xtr, ytr)
pi = permutation_importance(gb, Xte, yte, n_repeats=5, random_state=42, scoring="r2")
imp = pd.Series(pi.importances_mean, index=feats).sort_values()
names = {"log_carat": "carat (log)", "depth": "depth %", "table": "table %", "cut_ord": "cut", "color_ord": "colour", "clarity_ord": "clarity"}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
ax[0].barh([names[i] for i in imp.index], imp.values, color=BLUE); ax[0].set_title("Permutation importance"); ax[0].set_xlabel("Drop in R\u00b2 when shuffled")
pd1 = partial_dependence(gb, Xte, ["log_carat"], grid_resolution=40, method="brute")
ax[1].plot(np.exp(pd1["grid_values"][0]), np.exp(pd1["average"][0]), color=GREEN, lw=2); ax[1].set_title("Partial dependence: carat"); ax[1].set_xlabel("Carat"); ax[1].set_ylabel("Predicted price (USD)")
pd2 = partial_dependence(gb, Xte, ["clarity_ord"], grid_resolution=8, method="brute")
ax[2].plot(range(8), np.exp(pd2["average"][0]), marker="o", color="#805ad5", lw=2); ax[2].set_xticks(range(8)); ax[2].set_xticklabels(CLARITY, fontsize=8); ax[2].set_title("Partial dependence: clarity"); ax[2].set_xlabel("Clarity (worst to best)"); ax[2].set_ylabel("Predicted price (USD)")
plt.tight_layout(); plt.savefig("figures/v11_importance_pdp.png"); plt.close()
say("importance", imp.round(3).to_dict()); say("PDP clarity", np.exp(pd2["average"][0]).round(0).tolist())
pc = np.exp(pd1["average"][0]); gv = np.exp(pd1["grid_values"][0]); say("PDP carat points", [(round(float(a), 2), round(float(b))) for a, b in list(zip(gv, pc))[::8]])

# ---------- Fig 12  ACTUAL vs PREDICTED ----------
res = {}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for a, m, t, k in zip(ax, [lr, gb], ["Linear regression", "Gradient boosting"], ["lin", "gb"]):
    pr = m.predict(Xte); act = np.exp(yte); prd = np.exp(pr)
    res[k] = (r2_score(yte, pr), np.sqrt(np.mean((yte - pr) ** 2)), mean_absolute_error(act, prd), np.median(np.abs(prd - act) / act) * 100)
    h = a.hexbin(act, prd, gridsize=55, xscale="log", yscale="log", bins="log", cmap="viridis", mincnt=1)
    lim = [act.min(), act.max()]; a.plot(lim, lim, color=RED, ls="--", lw=1.2); a.set_xlabel("Actual price (USD)"); a.set_ylabel("Predicted price (USD)")
    a.set_title(f"{t}: R\u00b2(log) = {res[k][0]:.3f}")
plt.colorbar(h, ax=ax, label="Count (log scale)", shrink=.85)
plt.savefig("figures/v12_actual_vs_pred.png", bbox_inches="tight"); plt.close()
say("holdout R2(log), RMSE(log), MAE($), median APE %:", {k: [round(x, 3) for x in v] for k, v in res.items()})


# ---------- Fig 13  ROUND-CARAT JUMPS (line + bar) ----------
d = df[(df.carat >= 0.3) & (df.carat <= 2.05)].copy(); d["c2"] = d.carat.round(2)
g = d.groupby("c2").agg(n=("price", "size"), med=("price", "median"))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].bar(g.index, g.n, width=0.008, color=BLUE); ax[0].set_title("Diamonds per carat weight (0.01 ct steps)"); ax[0].set_xlabel("Carat"); ax[0].set_ylabel("Count")
for v in [0.5, 1.0, 1.5, 2.0]: ax[0].axvline(v, color=RED, ls=":", lw=1)
gg = g[g.n >= 8]; ax[1].plot(gg.index, gg.med, color=GREEN, lw=1.2); ax[1].set_title("Median price per carat weight: jumps at round sizes"); ax[1].set_xlabel("Carat"); ax[1].set_ylabel("Median price (USD)")
for v in [0.5, 1.0, 1.5, 2.0]: ax[1].axvline(v, color=RED, ls=":", lw=1)
plt.tight_layout(); plt.savefig("figures/v13_round_carat.png"); plt.close()
say("count at 0.99/1.00/1.01:", {k: int(g.n.get(k, 0)) for k in [0.99, 1.0, 1.01]}, "| median", {k: float(g.med.get(k, np.nan)) for k in [0.99, 1.0, 1.01, 0.49, 0.5, 0.51, 1.49, 1.5, 1.51]})
say("share of diamonds at exactly 0.30/0.50/0.70/1.00/1.01 ct:", {k: round(float(g.n.get(k, 0)) / len(d) * 100, 1) for k in [0.3, 0.5, 0.7, 1.0, 1.01]})

# ---------- Interactive Plotly (bonus) ----------
import plotly.express as px
sp = df.sample(6000, random_state=5)
fx = px.scatter(sp, x="carat", y="price", color="clarity", symbol="cut", log_x=True, log_y=True, hover_data=["color", "depth", "table"], category_orders={"clarity": CLARITY, "cut": CUT}, title="Diamonds: price vs carat (hover, zoom, click legend to filter)")
fx.update_traces(marker=dict(size=5, opacity=.6)); fx.write_html("interactive_explorer.html", include_plotlyjs="cdn")
open("viz_log.txt", "w").write("\n".join(log))
