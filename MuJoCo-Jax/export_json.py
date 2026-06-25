import pickle
import json
import numpy as np
import jax     
import flax     

with open("spider_policy_best.pkl", "rb") as f:
    raw_params = pickle.load(f)


if hasattr(raw_params, 'unfreeze'):
    raw_params = raw_params.unfreeze()

p = raw_params['params']

weights = {
    # --- Trunk Condiviso ---
    'W1': np.array(p['Dense_0']['kernel']).T.tolist(),
    'b1': np.array(p['Dense_0']['bias']).tolist(),
    
    'W2': np.array(p['Dense_1']['kernel']).T.tolist(),
    'b2': np.array(p['Dense_1']['bias']).tolist(),

    'W3': np.array(p['Dense_2']['kernel']).T.tolist(),
    'b3': np.array(p['Dense_2']['bias']).tolist(),

    'W4': np.array(p['Dense_3']['kernel']).T.tolist(),
    'b4': np.array(p['Dense_3']['bias']).tolist()
}

with open("spider_weights.json", "w") as f:
    json.dump(weights, f)

print("[OK] Pesi pronti per Simulink in 'spider_weights.json'.")