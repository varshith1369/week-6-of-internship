"""Week 2 - Data Cleaning and Pre-Processing (Diamonds dataset)
Run:  python clean.py      (needs diamonds.csv in the same folder)
Outputs: diamonds_clean.csv, diamonds_model_ready.csv, figures/*.png, cleaning_log.txt
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sns.set_theme(style="whitegrid")
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

CUT = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR = list("JIHGFED")                       # J (worst) -> D (best)
CLARITY = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
DIMS = ["x", "y", "z"]

# ---------- 1. INSPECTION ----------
raw = pd.read_csv("diamonds.csv")
df = raw.copy()
say("== 1. INSPECTION ==")
say("shape", raw.shape, "| nulls:", int(raw.isna().sum().sum()), "| exact duplicates:", int(raw.duplicated().sum()))
say("zeros per dim:", {c: int((raw[c] == 0).sum()) for c in DIMS}, "| rows with any zero:", int((raw[DIMS] == 0).any(axis=1).sum()))
for c in ["cut", "color", "clarity"]:
    say(c, "unique:", sorted(raw[c].unique()), "| whitespace/case issues:", int((raw[c] != raw[c].str.strip().str.title()).sum()) if c != "clarity" and c != "color" else int((raw[c] != raw[c].str.strip()).sum()))
raw_calc_depth = 2 * raw.z / (raw.x + raw.y) * 100
ok = (raw[DIMS] > 0).all(axis=1)
mism = ok & ((raw.depth - raw_calc_depth).abs() > 1)
say("depth inconsistent with x,y,z (|diff|>1pt) among valid rows:", int(mism.sum()))
say("table range", raw.table.min(), raw.table.max(), "| depth range", raw.depth.min(), raw.depth.max())

def iqr_out(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())
say("IQR outliers raw:", {c: iqr_out(raw[c]) for c in ["carat", "depth", "table", "price", "x", "y", "z"]})

fig, ax = plt.subplots(1, 4, figsize=(12, 3.4))
for a, c in zip(ax, DIMS + ["table"]):
    sns.boxplot(y=raw[c], ax=a, color="#fc8181"); a.set_title(f"{c} (raw)")
plt.tight_layout(); plt.savefig("figures/w2_f1_raw_boxplots.png", dpi=150); plt.close()

# ---------- 2. DUPLICATES ----------
say("\n== 2. DUPLICATES ==")
n0 = len(df); df = df.drop_duplicates(keep="first").reset_index(drop=True)
say("removed", n0 - len(df), "rows ->", len(df))

# ---------- 3. DISGUISED MISSING VALUES ----------
say("\n== 3. MISSING VALUES ==")
bad = (df[DIMS] == 0)                                        # zero = impossible
bad["y"] = bad["y"] | (df.y > 20)                            # > 20 mm = data-entry error
bad["z"] = bad["z"] | (df.z > 20)
say("impossible entries per dim:", {c: int(bad[c].sum()) for c in DIMS}, "| rows affected:", int(bad.any(axis=1).sum()))
say("rows with 2+ bad dims:", int((bad.sum(axis=1) >= 2).sum()), "| rows with all 3 bad:", int((bad.sum(axis=1) == 3).sum()))
shifted = (df[DIMS] > 20).any(axis=1)                      # a 31.8 / 58.9 mm value => the row's dimensions are shifted/unreliable
say("rows with a dimension > 20 mm (all 3 dimensions re-imputed):", int(shifted.sum()))
for c in DIMS: bad[c] = bad[c] | shifted
flagged = bad.any(axis=1)
say(df.loc[flagged, ["carat", "price"] + DIMS].round(2).to_string())
df["dim_imputed"] = flagged.astype(int)
for c in DIMS: df.loc[bad[c], c] = np.nan
say("NaN count after conversion:", df[DIMS].isna().sum().to_dict())

# ---------- 3b. COMPARE IMPUTATION METHODS (masking experiment) ----------
good = df.loc[~flagged].copy()
rng = np.random.default_rng(42)
test_idx = good.sample(5000, random_state=1).index
ref = good.drop(index=test_idx)                        # reference pool never sees the hidden truth

def impute(frame, method, ref):
    f = frame.copy()
    if method == "mean":   return f.fillna(ref[DIMS].mean())
    if method == "median": return f.fillna(ref[DIMS].median())
    if method == "regression":                         # log(dim) ~ log(carat)
        for c in DIMS:
            m = LinearRegression().fit(np.log(ref[["carat"]]), np.log(ref[c]))
            idx = f[c].isna()
            if idx.any(): f.loc[idx, c] = np.exp(m.predict(np.log(f.loc[idx, ["carat"]])))
        return f
    if method == "knn":                                # 5 nearest neighbours on log(carat), x, y, z (standardised)
        cols = ["carat"] + DIMS
        pool = pd.concat([ref[cols], f[cols]]); pool["carat"] = np.log(pool["carat"])
        sc = StandardScaler().fit(pool)
        Z = sc.transform(pool)
        Zi = KNNImputer(n_neighbors=5).fit_transform(Z)
        out = sc.inverse_transform(Zi)[len(ref):, 1:]
        f[DIMS] = out
        return f

def experiment(n_hidden):
    rows = []
    for m in ["mean", "median", "regression", "knn"]:
        errs = []
        t = good.loc[test_idx].copy(); truth = t[DIMS].copy()
        mask = pd.DataFrame(False, index=t.index, columns=DIMS)
        if n_hidden == 1:
            for c in DIMS: mask[c] = rng.random(len(t)) < 0.33
        else:
            hide = rng.random(len(t)) < 0.33
            for c in DIMS: mask[c] = hide
        t = t.mask(mask.reindex(columns=t.columns, fill_value=False))
        filled = impute(t, m, ref)
        e = (filled[DIMS] - truth)[mask]
        e = e.stack()
        rows.append((m, np.sqrt((e ** 2).mean()), e.abs().mean()))
    return pd.DataFrame(rows, columns=["method", f"RMSE_{n_hidden}", f"MAE_{n_hidden}"])
imp = experiment(1).merge(experiment(3), on="method").round(3)
imp.columns = ["method", "RMSE_one_dim_hidden", "MAE_one_dim_hidden", "RMSE_all_dims_hidden", "MAE_all_dims_hidden"]
say("\nImputation masking experiment (hide known values in 5,000 rows, then predict them):\n" + imp.to_string(index=False))
imp.to_csv("imputation_comparison.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
imp.set_index("method")[["RMSE_one_dim_hidden", "MAE_one_dim_hidden"]].plot.bar(ax=ax[0], color=["#2b6cb0", "#68d391"]); ax[0].set_title("One dimension missing"); ax[0].set_ylabel("error (mm)"); ax[0].tick_params(axis="x", rotation=0)
imp.set_index("method")[["RMSE_all_dims_hidden", "MAE_all_dims_hidden"]].plot.bar(ax=ax[1], color=["#2b6cb0", "#68d391"]); ax[1].set_title("All three dimensions missing"); ax[1].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig("figures/w2_f2_imputation.png", dpi=150); plt.close()

best = imp.sort_values("RMSE_one_dim_hidden").iloc[0].method
say("chosen method:", best)
before = df.copy()
df = impute(df, best, good)
say("remaining NaN:", int(df[DIMS].isna().sum().sum()))
say(df.loc[flagged, ["carat"] + DIMS].round(2).to_string())

# ---------- 4. INCONSISTENCIES + OUTLIERS ----------
say("\n== 4. INCONSISTENCIES / OUTLIERS ==")
df["depth_calc"] = 2 * df.z / (df.x + df.y) * 100
df["depth_mismatch"] = ((df.depth - df.depth_calc).abs() > 1).astype(int)
say("depth_mismatch after imputation:", int(df.depth_mismatch.sum()), "(reported depth vs computed from x,y,z, >1 point apart)")
say("table > 73:", int((df.table > 73).sum()), "| table < 50:", int((df.table < 50).sum()), "| depth<50 or >75:", int(((df.depth < 50) | (df.depth > 75)).sum()))
say("price == 0 or negative:", int((df.price <= 0).sum()), "| carat <= 0:", int((df.carat <= 0).sum()))
say("IQR outliers after cleaning:", {c: iqr_out(df[c]) for c in ["carat", "depth", "table", "price", "x", "y", "z"]})
z = (df[["carat", "price"]] - df[["carat", "price"]].mean()) / df[["carat", "price"]].std()
say("|z|>3 :", {"carat": int((z.carat.abs() > 3).sum()), "price": int((z.price.abs() > 3).sum())})
# winsorise only extreme table/depth (physically implausible), flag + keep price/carat
tbl_hi, tbl_lo = 73, 50
df["table_flag"] = ((df.table > tbl_hi) | (df.table < tbl_lo)).astype(int)
n_tbl = int(df.table_flag.sum())
df["table"] = df.table.clip(tbl_lo, tbl_hi)
say("table values capped to [50,73]:", n_tbl)
df["carat_gt3"] = (df.carat > 3).astype(int)
say("carat > 3 kept and flagged:", int(df.carat_gt3.sum()))

# plot before/after
fig, ax = plt.subplots(2, 3, figsize=(11, 6))
for j, c in enumerate(DIMS):
    ax[0, j].scatter(before.carat, before[c], s=1, alpha=.3, color="#e53e3e"); ax[0, j].set_title(f"{c} vs carat - before")
    ax[1, j].scatter(df.carat, df[c], s=1, alpha=.3, color="#2f855a"); ax[1, j].set_title(f"{c} vs carat - after")
plt.tight_layout(); plt.savefig("figures/w2_f3_before_after.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 4, figsize=(12, 3.4))
for a, c in zip(ax, DIMS + ["table"]):
    sns.boxplot(y=df[c], ax=a, color="#68d391"); a.set_title(f"{c} (cleaned)")
plt.tight_layout(); plt.savefig("figures/w2_f4_clean_boxplots.png", dpi=150); plt.close()

# ---------- 5. FEATURE ENGINEERING ----------
say("\n== 5. FEATURE ENGINEERING ==")
df["log_price"] = np.log(df.price); df["log_carat"] = np.log(df.carat)
df["volume"] = df.x * df.y * df.z
df["price_per_carat"] = df.price / df.carat
df["xy_ratio"] = df.x / df.y
df["size_class"] = pd.cut(df.carat, [0, .4, .7, 1.0, 1.5, 6], labels=["small", "medium", "large", "very_large", "exceptional"])
df["cut_ord"] = df.cut.map({k: i for i, k in enumerate(CUT)})
df["color_ord"] = df.color.map({k: i for i, k in enumerate(COLOR)})
df["clarity_ord"] = df.clarity.map({k: i for i, k in enumerate(CLARITY)})
df["quality_score"] = df.cut_ord + df.color_ord + df.clarity_ord
say("skew price", round(df.price.skew(), 2), "-> log_price", round(df.log_price.skew(), 2), "| carat", round(df.carat.skew(), 2), "-> log_carat", round(df.log_carat.skew(), 2))
say("corr(log_carat, log_price) =", round(df.log_carat.corr(df.log_price), 3), "| corr(carat, price) =", round(df.carat.corr(df.price), 3))
say("corr(volume, carat) =", round(df.volume.corr(df.carat), 3))
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
sns.histplot(df.price, bins=50, ax=ax[0], color="#2b6cb0"); ax[0].set_title("price")
sns.histplot(df.log_price, bins=50, ax=ax[1], color="#2f855a"); ax[1].set_title("log(price)")
ax[2].scatter(df.log_carat, df.log_price, s=1, alpha=.2, color="#2b6cb0"); ax[2].set_title("log(carat) vs log(price)")
plt.tight_layout(); plt.savefig("figures/w2_f5_transform.png", dpi=150); plt.close()

num = ["carat", "depth", "table", "x", "y", "z", "volume"]
std = pd.DataFrame(StandardScaler().fit_transform(df[num]), columns=[c + "_std" for c in num])
mm = pd.DataFrame(MinMaxScaler().fit_transform(df[["price", "carat"]]), columns=["price_mm", "carat_mm"])
say("standardised means ~0:", std.mean().round(3).abs().max(), "| std ~1:", std.std().round(2).unique().tolist())
oh = pd.get_dummies(df[["cut", "color", "clarity"]], drop_first=True, dtype=int)
say("one-hot columns:", oh.shape[1])
model_ready = pd.concat([df[["log_price", "price"]], std, mm, df[["cut_ord", "color_ord", "clarity_ord", "quality_score", "xy_ratio", "log_carat"]], oh], axis=1)
say("model-ready shape", model_ready.shape, "| nulls", int(model_ready.isna().sum().sum()))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
df[num].boxplot(ax=ax[0]); ax[0].set_title("Before scaling")
std.boxplot(ax=ax[1]); ax[1].set_title("After standardisation"); ax[1].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig("figures/w2_f6_scaling.png", dpi=150); plt.close()

# ---------- 6. SAVE ----------
df.to_csv("diamonds_clean.csv", index=False); model_ready.to_csv("diamonds_model_ready.csv", index=False)
say("\n== 6. FINAL ==")
say("rows raw", len(raw), "-> clean", len(df), "| columns raw", raw.shape[1], "-> clean", df.shape[1])
say(df[["carat", "depth", "table", "price", "x", "y", "z"]].describe().round(2).to_string())
say("final nulls", int(df.drop(columns=["size_class"]).isna().sum().sum()), "| size_class nulls", int(df.size_class.isna().sum()))
open("cleaning_log.txt", "w").write("\n".join(log))
