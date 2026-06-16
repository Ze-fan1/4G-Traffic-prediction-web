#!/usr/bin/env python
"""
Batch run: 6 new models + re-run IBM TTM with scaled space fix.
Non-interactive, runs all models in sequence.
"""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from run import run_traditional_single, run_dl_model, run_ttm

print("=" * 60)
print("  BATCH MODEL RUNNER")
print("=" * 60)

# Step 1: Traditional models (fast, no training)
print("\n### Step 1: Traditional models ###")
try:
    run_traditional_single('AutoAR')
except Exception as e:
    print(f"[FAIL] AutoAR: {e}")

try:
    run_traditional_single('LinearRegression')
except Exception as e:
    print(f"[FAIL] LinearRegression: {e}")

# Step 2: DL models (10 epochs each, lightweight params)
print("\n### Step 2: DL models ###")
for model_name in ['Informer', 'LightTS', 'TSMixer', 'SCINet']:
    try:
        run_dl_model(model_name)
    except Exception as e:
        print(f"[FAIL] {model_name}: {e}")

# Step 3: IBM TTM (re-run with scaled space save fix)
print("\n### Step 3: IBM TTM (scaled space) ###")
try:
    run_ttm()
except Exception as e:
    print(f"[FAIL] IBM TTM: {e}")

print("\n" + "=" * 60)
print("  BATCH COMPLETE")
print("=" * 60)
