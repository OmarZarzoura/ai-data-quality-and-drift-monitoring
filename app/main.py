from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd, numpy as np
from scipy.stats import ks_2samp

app = FastAPI(title="AI Data Quality & Drift Monitoring", version="1.0.0")
baseline = None

def psi(expected, actual, bins=10):
    expected = np.asarray(expected); actual = np.asarray(actual)
    qs = np.linspace(0, 1, bins+1)
    b = np.unique(np.quantile(expected, qs))
    def hist(x):
        h,_ = np.histogram(x, bins=b)
        return h / (len(x)+1e-9)
    e = hist(expected) + 1e-6
    a = hist(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))

class Frame(BaseModel):
    data: list[dict]

@app.post("/baseline")
def set_baseline(f: Frame):
    global baseline
    baseline = pd.DataFrame(f.data)
    return {"rows": len(baseline)}

@app.post("/monitor")
def monitor(f: Frame):
    if baseline is None:
        return {"error": "baseline not set"}
    current = pd.DataFrame(f.data)
    report = {}
    for col in baseline.columns:
        if col not in current.columns:
            report[col] = {"status":"missing in current"}
            continue
        x = pd.to_numeric(baseline[col], errors="coerce").dropna()
        y = pd.to_numeric(current[col], errors="coerce").dropna()
        if len(x) > 5 and len(y) > 5:
            ks = ks_2samp(x, y)
            ps = psi(x, y)
            drift = (ks.pvalue < 0.01) or (ps > 0.25)
            report[col] = {"ks_stat": float(ks.statistic), "p_value": float(ks.pvalue), "psi": ps, "drift": bool(drift)}
        else:
            report[col] = {"status": "insufficient numeric data"}
    return {"columns": report}
