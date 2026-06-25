"""
enjoy_spider.py — Teleoperazione con Data Logging per MATLAB
============================================================
Guida il robot col gamepad. Alla chiusura, genera 'telemetry.csv'.
"""

import mujoco
import mujoco.viewer
import numpy as np
import json
import time
import os
import pygame
import sys
import csv

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def load_policy(json_path="spider_weights.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    return {k: np.array(v) for k, v in data.items()}

def policy(weights, obs):
    """ Inferenza End-to-End numpy """
    x = np.dot(weights['W1'], obs) + weights['b1']
    x = elu(x)
    x = np.dot(weights['W2'], x) + weights['b2']
    x = elu(x)
    x = np.dot(weights['W3'], x) + weights['b3']
    x = elu(x)
    x = np.dot(weights['W4'], x) + weights['b4']
    return np.tanh(x)

class JoystickMapper:
    def __init__(self):
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"[INFO] Controller Rilevato: {self.joystick.get_name()}")
        else:
            print("[WARN] Nessun controller rilevato! Lo SpiderBot resterà fermo.")
            
    def get_commands(self):
        pygame.event.pump()
        if not self.joystick:
            return 0.0, 0.0, 0.0
            
        deadzone = 0.15
        
        raw_vx = self.joystick.get_axis(0) 
        raw_vy = -self.joystick.get_axis(1) 
        raw_wz = -self.joystick.get_axis(3) 
        
        vx = raw_vx if abs(raw_vx) > deadzone else 0.0
        vy = raw_vy if abs(raw_vy) > deadzone else 0.0
        wz = raw_wz if abs(raw_wz) > deadzone else 0.0
        
        cmd_vx = vx * 1.0
        cmd_vy = vy * 1.0
        cmd_wz = wz * 1.0
        
        return cmd_vx, cmd_vy, cmd_wz

def enjoy():
    print("[INFO] Caricamento pesi da 'spider_weights.json'...")
    try:
        weights = load_policy("spider_weights.json")
    except FileNotFoundError:
        print("[ERRORE] File dei pesi non trovato. Esegui prima 'python export_json.py'")
        return
    
    model = mujoco.MjModel.from_xml_path("spiderbot.xml")
    data = mujoco.MjData(model)
    
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.25 # Z_TARGET
    mujoco.mj_forward(model, data)
    
    joystick = JoystickMapper()
    
    # --- Inizializzazione Logger ---
    telemetry_data = []
    
    print("[INFO] Avvio simulazione. Guida il robot!")
    print("[INFO] Alla chiusura della finestra, i dati verranno salvati in 'telemetry.csv'.\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        render_fps = 60
        render_interval = 1.0 / render_fps
        last_render_time = time.time()
        
        real_start_time = time.time()
        sim_time = 0.0 
        
        try:
            while viewer.is_running():
                current_real_time = time.time() - real_start_time
                
                while sim_time < current_real_time:
                    cmd_vx, cmd_vy, cmd_wz = joystick.get_commands()
                    commands = np.array([cmd_vx, cmd_vy, cmd_wz])
                    
                    position = data.qpos[2:]  
                    velocity = data.qvel      
                    obs = np.concatenate([position, velocity, commands])
                    
                    action_bounded = policy(weights, obs)
                    ctrl = action_bounded * 2.5
                    data.ctrl[:] = np.clip(ctrl, -2.5, 2.5)
                    
                    mujoco.mj_step(model, data)
                    
                    # Logghiamo i dati ad ogni step di simulazione (es. 200 Hz)
                    real_vx, real_vy, real_wz = data.qvel[0], data.qvel[1], data.qvel[5]
                    telemetry_data.append([
                        sim_time, 
                        cmd_vx, cmd_vy, cmd_wz, 
                        real_vx, real_vy, real_wz
                    ])
                    
                    sim_time += model.opt.timestep

                # Rendering
                current_time = time.time()
                if (current_time - last_render_time) >= render_interval:
                    viewer.sync()
                    last_render_time = current_time
                    
                    sys.stdout.write(
                        f"\r🎯 CMD [X: {cmd_vx:>5.2f} | Y: {cmd_vy:>5.2f} | Yaw: {cmd_wz:>5.2f}]  ||  "
                        f"🤖 REAL [X: {real_vx:>5.2f} | Y: {real_vy:>5.2f} | Yaw: {real_wz:>5.2f}]"
                    )
                    sys.stdout.flush()
                
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n[INFO] Simulazione interrotta manualmente.")

    # --- Salvataggio su file CSV per MATLAB ---
    print("\n\n[INFO] Chiusura simulatore. Salvataggio telemetria in corso...")
    csv_filename = "telemetry.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["time", "cmd_vx", "cmd_vy", "cmd_wz", "real_vx", "real_vy", "real_wz"])
        writer.writerows(telemetry_data)
        
    print(f"[OK] Salvati {len(telemetry_data)} campioni in '{csv_filename}'. Pronto per MATLAB!")

if __name__ == "__main__":
    enjoy()