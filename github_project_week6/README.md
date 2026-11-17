# Diamond Price Analysis – End-to-End Data Analytics Project (Weeks 1–6)

A complete analytics workflow on the public Diamonds dataset (53,940 diamonds): exploration, cleaning, visualization, statistical testing, machine learning and synthesis.

## Key results
- Weight dominates price: log-log correlation 0.966, elasticity 1.68 (R² = 0.93 from carat alone)
- Controlling for carat: clarity IF +204% vs I1, colour J -40% vs D, Ideal cut +17.6% vs Fair
- Raw data makes Ideal cut look 36% cheaper than Fair (confounding by size); at equal carat it is +37%
- Depth and table have no practical effect (within ±2% per SD, equivalence test)
- Gradient boosting predicts price with median error 5.7% (MAE $270, 73% within ±10%) on 10,759 unseen diamonds
- Classification of high-value diamonds: accuracy 0.977, F1 0.952, ROC AUC 0.997

## Project structure
| File | Purpose |
|---|---|
| `diamonds.csv` | Raw data |
| `01_eda.py` | Week 1 – exploration |
| `02_clean.py` | Week 2 – cleaning, imputation experiment, feature engineering |
| `03_carat_control_preview.py` | Week 2 – carat confounding and non-linearity preview |
| `04_visualize.py` | Week 3 – 13 figures + interactive Plotly chart |
| `05_stats_tests.py` | Week 4 – 7 hypothesis tests with assumption checks |
| `06_ml_models.py` | Week 5 – model training, tuning, evaluation |
| `run_all.py` | Runs everything in order (~3 minutes on one CPU core) |
| `figures/`, `*_log.txt`, `*.csv` | Generated outputs |
| `reports/` | Word reports for Weeks 1–6 (Week 6 is the capstone report) |

## Run
```
pip install -r requirements.txt
python run_all.py
```
