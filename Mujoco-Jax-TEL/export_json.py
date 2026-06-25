"""
export_json.py — Esportatore Pesi Agnostico 
===========================================
Converte i parametri flax/jax in numpy arrays (JSON)
per inferenza esterna o C++/Simulink. Supporta dimensioni dinamiche.
"""

import pickle
import json
import numpy as np

# Usa "spider_policy_final.pkl" o "spider_policy_best.pkl"
FILE_PATH = "spider_policy_best.pkl"

with open(FILE_PATH, "rb") as f:
    raw_params = pickle.load(f)

if hasattr(raw_params, 'unfreeze'):
    raw_params = raw_params.unfreeze()

p = raw_params['params']

weights = {
    # --- Trunk Condiviso (Adattivo all'OBS_DIM) ---
    'W1': np.array(p['Dense_0']['kernel']).T.tolist(),
    'b1': np.array(p['Dense_0']['bias']).tolist(),
    
    'W2': np.array(p['Dense_1']['kernel']).T.tolist(),
    'b2': np.array(p['Dense_1']['bias']).tolist(),
    
    # --- Actor Branch ---
    'W3': np.array(p['Dense_2']['kernel']).T.tolist(),
    'b3': np.array(p['Dense_2']['bias']).tolist(),
    
    # --- Actor Output (12 DOF) ---
    'W4': np.array(p['Dense_3']['kernel']).T.tolist(),
    'b4': np.array(p['Dense_3']['bias']).tolist()
}

with open("spider_weights.json", "w") as f:
    json.dump(weights, f)

print(f"[OK] Pesi estratti da {FILE_PATH} e salvati in 'spider_weights.json'.")