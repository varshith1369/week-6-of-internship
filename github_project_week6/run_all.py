"""Run the complete analysis pipeline (Weeks 1-5) in order.
Usage:  python run_all.py
Requires diamonds.csv in this folder. Total runtime is roughly 10 minutes on one CPU core."""
import subprocess, sys, time, os
os.makedirs("figures", exist_ok=True)
STEPS = [("01_eda.py", "Week 1  exploratory data analysis"),
         ("02_clean.py", "Week 2  cleaning, imputation, feature engineering"),
         ("03_carat_control_preview.py", "Week 2  carat control and non-linearity preview"),
         ("04_visualize.py", "Week 3  visualizations"),
         ("05_stats_tests.py", "Week 4  statistical tests"),
         ("06_ml_models.py", "Week 5  machine learning")]
t0 = time.time()
for script, label in STEPS:
    t = time.time(); print(f"\n=== {label} ({script}) ===", flush=True)
    r = subprocess.run([sys.executable, script], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:]); sys.exit(f"FAILED: {script}")
    print(f"done in {time.time() - t:.0f}s", flush=True)
print(f"\nAll steps finished in {(time.time() - t0) / 60:.1f} minutes. Outputs: figures/, *.csv, *_log.txt, interactive_explorer.html")
