import mujoco
import mujoco.viewer
import numpy as np
import json
import time
import csv
import os

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def load_policy(json_path="spider_weights.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    return {k: np.array(v) for k, v in data.items()}

def policy(weights, obs):
    """ Inferenza pura Numpy della rete """
    x = np.dot(weights['W1'], obs) + weights['b1']
    x = elu(x)
    x = np.dot(weights['W2'], x) + weights['b2']
    x = elu(x)
    x = np.dot(weights['W3'], x) + weights['b3']
    x = elu(x)
    # Output finale
    x = np.dot(weights['W4'], x) + weights['b4']
    return np.tanh(x)

def get_obs(data):
    position = data.qpos[2:]  
    velocity = data.qvel      
    return np.concatenate([position, velocity])

def enjoy():
    print("[INFO] Caricamento pesi da JSON...")
    weights = load_policy("spider_weights.json")
    
    model = mujoco.MjModel.from_xml_path("spiderbot.xml")
    data = mujoco.MjData(model)
    
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.25 # Z_TARGET
    mujoco.mj_forward(model, data)
    
    # Buffer per raccogliere i dati
    history_pos = []
    history_vel = []
    history_act = []
    history_time = []
    
    print("[INFO] Avvio visualizzazione 3D. Premi ESC per chiudere.")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        render_fps = 60
        render_interval = 1.0 / render_fps
        last_render_time = time.time()
        
        real_start_time = time.time()
        sim_time = 0.0 
        
        while viewer.is_running():
            current_real_time = time.time() - real_start_time
            
            while sim_time < current_real_time:
                obs = get_obs(data)
                
                # Inferenza Policy
                action_bounded = policy(weights, obs)
                
                # Salva i dati nei buffer (usando .copy() per evitare riferimenti al buffer di mujoco)
                history_pos.append(data.qpos.copy())
                history_vel.append(data.qvel.copy())
                history_act.append(action_bounded.copy())
                history_time.append(sim_time)
                
                # Applicazione controllo
                ctrl = action_bounded * 1.5
                data.ctrl[:] = np.clip(ctrl, -1.5, 1.5)
                
                mujoco.mj_step(model, data)
                sim_time += model.opt.timestep

            # Rendering sincronizzato
            current_time = time.time()
            if (current_time - last_render_time) >= render_interval:
                viewer.sync()
                last_render_time = current_time
            
            time.sleep(0.001) # Piccolo sleep per non saturare la CPU

    # Salvataggio finale unico (molto veloce)
    np.savez('sim_data.npz', 
             pos=np.array(history_pos), 
             vel=np.array(history_vel), 
             act=np.array(history_act),
             time=np.array(history_time))
             
    print(f"[INFO] Simulazione terminata. Dati salvati in 'sim_data.npz' ({len(history_time)} step)")

if __name__ == "__main__":
    enjoy()