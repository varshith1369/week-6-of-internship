"""Preview of the Week 3/4 analysis plan on the CLEANED data:
(1) how carat is controlled for, (2) how non-linear effects will be tested.
Run after clean.py:  python carat_control_preview.py
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
sns.set_theme(style="whitegrid")
log=[]
def say(*a):
    s=" ".join(str(x) for x in a); print(s); log.append(s)
df = pd.read_csv("diamonds_clean.csv")
CUT=["Fair","Good","Very Good","Premium","Ideal"]
df["cut"]=pd.Categorical(df.cut,CUT,ordered=False)
df["color"]=pd.Categorical(df.color,list("DEFGHIJ")); df["clarity"]=pd.Categorical(df.clarity,["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])

say("== A. CONFOUNDING OF CARAT ==")
say("mean carat by cut:\n", df.groupby("cut",observed=True).carat.mean().round(2).to_string())
m0 = smf.ols("log_price ~ C(cut, Treatment('Fair'))", df).fit()
m1 = smf.ols("log_price ~ log_carat + C(cut, Treatment('Fair'))", df).fit()
c0 = m0.params.filter(like="cut").round(3); c1 = m1.params.filter(like="cut").round(3)
cmp = pd.DataFrame({"cut":CUT[1:], "without_carat":c0.values, "with_carat":c1.values})
cmp["pct_effect_without"]=(np.exp(cmp.without_carat)-1).mul(100).round(1)
cmp["pct_effect_with"]=(np.exp(cmp.with_carat)-1).mul(100).round(1)
say("Cut effect vs Fair on log(price), without and with carat control:\n", cmp.to_string(index=False))
cmp.to_csv("carat_control_cut_effects.csv",index=False)

# stratified by size band
df["band"]=pd.cut(df.carat,[0,.5,1,1.5,6],labels=["<=0.5","0.5-1","1-1.5",">1.5"])
tab=df.pivot_table(index="band",columns="cut",values="price",aggfunc="median",observed=True).round(0)
say("Median price by carat band x cut:\n", tab.to_string())
tab.to_csv("carat_band_by_cut_median_price.csv")

say("\n== B. COLLINEARITY (VIF) ==")
X = df[["log_carat","depth","table","x","y","z"]].copy(); X = sm.add_constant(X)
v1=pd.Series([variance_inflation_factor(X.values,i) for i in range(1,X.shape[1])],index=X.columns[1:]).round(1)
X2 = sm.add_constant(df[["log_carat","depth","table"]])
v2=pd.Series([variance_inflation_factor(X2.values,i) for i in range(1,X2.shape[1])],index=X2.columns[1:]).round(1)
say("VIF with x,y,z:\n", v1.to_string()); say("VIF after dropping x,y,z:\n", v2.to_string())

say("\n== C. NON-LINEARITY TESTS ==")
full_lin = smf.ols("log_price ~ log_carat + C(cut) + C(color) + C(clarity)", df).fit()
quad = smf.ols("log_price ~ log_carat + I(log_carat**2) + C(cut) + C(color) + C(clarity)", df).fit()
inter = smf.ols("log_price ~ log_carat * C(clarity) + C(cut) + C(color)", df).fit()
say("R2 linear", round(full_lin.rsquared,4), "| quadratic", round(quad.rsquared,4), "| carat x clarity", round(inter.rsquared,4))
say("AIC linear", round(full_lin.aic), "| quadratic", round(quad.aic), "| interaction", round(inter.aic))
a1=anova_lm(full_lin,quad); a2=anova_lm(full_lin,inter)
say("F-test quadratic term: F =", round(a1.F.iloc[1],1), "p =", f"{a1['Pr(>F)'].iloc[1]:.3g}")
say("F-test carat x clarity: F =", round(a2.F.iloc[1],1), "p =", f"{a2['Pr(>F)'].iloc[1]:.3g}")
say("quadratic coef on log_carat^2:", round(quad.params["I(log_carat ** 2)"],3))
# raw (non-log) carat linear vs log
raw_lin = smf.ols("price ~ carat + C(cut) + C(color) + C(clarity)", df).fit()
say("R2 raw-price linear model", round(raw_lin.rsquared,4), "(vs log-log", round(full_lin.rsquared,4), ")")
# Spearman vs Pearson
from scipy import stats
say("Pearson carat-price", round(df.carat.corr(df.price),3), "| Spearman", round(stats.spearmanr(df.carat,df.price)[0],3))

say("\n== D. LINEAR vs FLEXIBLE MODEL (5-fold CV, target log_price) ==")
feats=["log_carat","depth","table","cut_ord","color_ord","clarity_ord"]
kf=KFold(5,shuffle=True,random_state=42)
r_lin=cross_val_score(LinearRegression(),df[feats],df.log_price,cv=kf,scoring="neg_root_mean_squared_error").mean()*-1
r_gb=cross_val_score(HistGradientBoostingRegressor(random_state=42),df[feats],df.log_price,cv=kf,scoring="neg_root_mean_squared_error").mean()*-1
q_lin=cross_val_score(LinearRegression(),df[feats],df.log_price,cv=kf,scoring="r2").mean()
q_gb=cross_val_score(HistGradientBoostingRegressor(random_state=42),df[feats],df.log_price,cv=kf,scoring="r2").mean()
say("Linear: RMSE(log) =", round(r_lin,4), "R2 =", round(q_lin,4)); say("Gradient boosting: RMSE(log) =", round(r_gb,4), "R2 =", round(q_gb,4))

# figures
fig,ax=plt.subplots(1,3,figsize=(12,3.6))
sns.pointplot(data=df,x="cut",y="carat",order=CUT,ax=ax[0],color="#2b6cb0",errorbar=None); ax[0].set_title("Mean carat by cut (confounder)"); ax[0].tick_params(axis="x",rotation=25)
w=cmp.melt(id_vars="cut",value_vars=["pct_effect_without","pct_effect_with"],var_name="model",value_name="pct")
w["model"]=w.model.map({"pct_effect_without":"without carat","pct_effect_with":"with carat"})
sns.barplot(data=w,x="cut",y="pct",hue="model",ax=ax[1],palette=["#fc8181","#68d391"]); ax[1].set_title("Cut effect vs Fair on price (%)"); ax[1].set_ylabel("%"); ax[1].tick_params(axis="x",rotation=25); ax[1].axhline(0,color="k",lw=.8); ax[1].set_ylim(-45,62); ax[1].legend(loc="upper left",fontsize=7,ncol=2,title=None)
samp=df.sample(6000,random_state=1)
sns.regplot(data=samp,x="log_carat",y="log_price",lowess=True,scatter_kws={"s":2,"alpha":.2},line_kws={"color":"red"},ax=ax[2],color="#2b6cb0"); ax[2].set_title("LOWESS curve: log carat vs log price")
plt.tight_layout(); plt.savefig("figures/w2_f7_carat_control.png",dpi=150); plt.close()
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
for a,mod,t in zip(ax,[full_lin,quad],["Linear in log(carat)","Adding log(carat)^2"]):
    a.scatter(mod.fittedvalues[samp.index],mod.resid[samp.index],s=2,alpha=.25,color="#2b6cb0"); a.axhline(0,color="red"); a.set_title("Residuals: "+t); a.set_xlabel("fitted log price"); a.set_ylabel("residual")
plt.tight_layout(); plt.savefig("figures/w2_f8_residuals.png",dpi=150); plt.close()
open("carat_control_log.txt","w").write("\n".join(log))
