"""
spider_env.py — SpiderBotEnv v7 (Omnidirectional Goal-Conditioned RL)
=====================================================================
Rete Neurale condizionata dai comandi joystick.
Input (38D): Osservazioni (35D) + Comandi Target (3D: vx, vy, wz)
Output (12D): 12 target angolari [-2.5, 2.5] radianti

Modifiche: 
- Reward di tracking unificata per costringere l'aderenza a TUTTI gli assi.
- Penalità attiva per deriva (movimenti su assi non richiesti).
"""

import jax
import jax.numpy as jnp
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import mujoco

OBS_DIM = 38  # 17 Pos + 18 Vel + 3 Comandi (vx, vy, wz)
ACT_DIM = 12
MAX_STEPS = 1000

Z_TARGET = 0.25

class SpiderBotEnv(PipelineEnv):

    def __init__(self, xml_path: str = "spiderbot.xml", **kwargs):
        mj_model = mujoco.MjModel.from_xml_path(xml_path)

        mj_model.opt.solver        = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations    = 6
        mj_model.opt.ls_iterations = 6

        self.mj_model = mj_model
        sys = mjcf.load_model(mj_model)

        super().__init__(sys, backend='mjx', n_frames=4, **kwargs)

    def _get_obs(self, pipeline_state, commands) -> jnp.ndarray:
        position = pipeline_state.qpos[2:]   # (17,)
        velocity = pipeline_state.qvel       # (18,)
        # Concateniamo i comandi allo stato cinematico
        return jnp.concatenate([position, velocity, commands])

    def reset(self, rng: jnp.ndarray) -> State:
        rng, r1, r2, r_cmd = jax.random.split(rng, 4)

        noise_pos = jax.random.uniform(r1, (self.mj_model.nq,), minval=-0.04, maxval=0.04)
        qpos = jnp.array(self.mj_model.qpos0)
        qpos = qpos.at[2].set(Z_TARGET)
        
        qpos = qpos + noise_pos * 0.02
        
        q = qpos[3:7]
        qpos = qpos.at[3:7].set(q / jnp.linalg.norm(q))

        qvel = jax.random.uniform(r2, (self.mj_model.nv,), minval=-0.01, maxval=0.01)

        # Inizializza target casuali per l'episodio
        cmd_vx = jax.random.uniform(r_cmd, minval=-1.0, maxval=1.0) # Avanzamento (X)
        cmd_vy = jax.random.uniform(r_cmd, minval=-0.5, maxval=0.5) # Passo laterale (Y)
        cmd_wz = jax.random.uniform(r_cmd, minval=-1.0, maxval=1.0) # Rotazione (Yaw)
        commands = jnp.array([cmd_vx, cmd_vy, cmd_wz])

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, commands)

        zero = jnp.float32(0.0)
        metrics = {'reward': zero, 'track_error': zero}
        
        info = {
            'rng': rng, 
            'step': jnp.int32(0),
            'commands': commands,
            'last_action': jnp.zeros(ACT_DIM)
        }
        
        return State(pipeline_state, obs, zero, zero, metrics, info)

    def step(self, state: State, action: jnp.ndarray) -> State:
        rng, r_act, r_obs, r_cmd = jax.random.split(state.info['rng'], 4)
        step_n = state.info['step'] + 1

        # --- CURRICULUM: Ricampiona i comandi ogni 200 step per generalizzare ---
        cond_resample = (step_n % 200 == 0)
        new_cmd_vx = jax.random.uniform(r_cmd, minval=-1.0, maxval=1.0)
        new_cmd_vy = jax.random.uniform(r_cmd, minval=-1.0, maxval=1.0)
        new_cmd_wz = jax.random.uniform(r_cmd, minval=-1.0, maxval=1.0)
        new_commands = jnp.array([new_cmd_vx, new_cmd_vy, new_cmd_wz])
        
        commands = jnp.where(cond_resample, new_commands, state.info['commands'])

        # --- 1. CONTROLLO DIRETTO (Policy -> Giunti) ---
        ctrl = jnp.tanh(action) * 2.5
        ctrl = ctrl + jax.random.normal(r_act, ctrl.shape) * 0.02 # Sim-to-Real noise
        ctrl = jnp.clip(ctrl, -2.5, 2.5)

        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)

        obs = self._get_obs(pipeline_state, commands)
        obs = obs + jax.random.normal(r_obs, obs.shape) * 0.003

        # --- 2. REWARD SHAPING (Omnidirectional Goal-Conditioned) ---
        vel_x = pipeline_state.qvel[0]   # Velocità longitudinale
        vel_y = pipeline_state.qvel[1]   # Velocità laterale
        vel_wz = pipeline_state.qvel[5]  # Velocità angolare Z (Yaw)
        
        cmd_vx, cmd_vy, cmd_wz = commands[0], commands[1], commands[2]

        # A. Tracking Unificato: Errore quadratico cumulativo
        # Usare un unico esponenziale impone che TUTTI gli assi siano vicini al target
        # per ottenere un punteggio alto (la moltiplicazione implicita delle probabilità).
        sigma_vel = 0.50
        err_sq = jnp.square(vel_x - cmd_vx) + jnp.square(vel_y - cmd_vy) + jnp.square(vel_wz - cmd_wz)
        tracking_reward = 3.0 * jnp.exp(-err_sq / sigma_vel)

        # B. Penalità per Deriva (Orthogonal/Unrequested Movement Penalty)
        # Penalizza la velocità misurata se il comando corrispondente è vicino a zero.
        # jnp.exp(-cmd^2 / 0.01) funge da "switch" morbido: vale ~1 se cmd è 0, scende a 0 se cmd != 0.
        sharpness = 0.01
        p_drift_x = jnp.exp(-jnp.square(cmd_vx) / sharpness) * jnp.square(vel_x)
        p_drift_y = jnp.exp(-jnp.square(cmd_vy) / sharpness) * jnp.square(vel_y)
        p_drift_wz = jnp.exp(-jnp.square(cmd_wz) / sharpness) * jnp.square(vel_wz)
        p_drift = 3.0 * (p_drift_x + p_drift_y + p_drift_wz)

        # C. Penalità Cinematiche ed Energetiche
        last_action = state.info['last_action']
        p_smooth = 0.2 * jnp.sum(jnp.square(action - last_action))
        p_energy = 0.02 * jnp.sum(jnp.square(ctrl)) # Risparmio energetico

        torso_z = pipeline_state.qpos[2]
        tilt = jnp.square(pipeline_state.qpos[4]) + jnp.square(pipeline_state.qpos[5])
        is_fallen = (torso_z < 0.1) | (torso_z > 0.50) | (tilt > 0.3)
        p_fall = jnp.where(is_fallen, -100.0, 0.0)

        joint_angles = pipeline_state.qpos[7:19]
        p_posture = 0.5 * jnp.mean(jnp.square(joint_angles))

        # D. Survival Reward
        # Incentiva l'agente a non suicidarsi deliberatamente per evitare penalità di energia.
        r_survival = 1.0
        
        total_reward = (
            r_survival
            + tracking_reward
            + p_fall
            - p_drift
            - p_posture
            - p_smooth
            - p_energy
        )

        done = jnp.where(is_fallen | (step_n >= MAX_STEPS), 1.0, 0.0)

        new_info = state.info.copy()
        new_info.update({
            'rng': rng, 
            'step': step_n,
            'commands': commands,
            'last_action': action
        })

        metrics = {
            'reward': total_reward,
            'track_error': jnp.sqrt(err_sq)
        }

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=total_reward,
            done=done,
            metrics=metrics,
            info=new_info,
        )