"""
spider_env.py — SpiderBotEnv v5 (Pure RL per CPU)
=================================================
La Rete Neurale agisce da controllore End-to-End.
Input: Osservazioni (35D)
Output: 12 target angolari [-2.5, 2.5] radianti
Reward: Solo velocità lungo l'asse X.
"""

import jax
import jax.numpy as jnp
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import mujoco

OBS_DIM = 35
ACT_DIM = 12
MAX_STEPS = 1000  # <--- Limite esplicito dell'episodio

Z_TARGET = 0.25

class SpiderBotEnv(PipelineEnv):

    def __init__(self, xml_path: str = "spiderbot.xml", **kwargs):
        mj_model = mujoco.MjModel.from_xml_path(xml_path)

        mj_model.opt.solver        = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations    = 6
        mj_model.opt.ls_iterations = 6

        self.mj_model = mj_model
        sys = mjcf.load_model(mj_model)

        # Rimosso il caricamento MATLAB. L'agente esplora da zero.
        super().__init__(sys, backend='mjx', n_frames=4, **kwargs)

    def _get_obs(self, pipeline_state) -> jnp.ndarray:
        position = pipeline_state.qpos[2:]   # (17,)
        velocity = pipeline_state.qvel       # (18,)
        return jnp.concatenate([position, velocity])

    def reset(self, rng: jnp.ndarray) -> State:
        rng, r1, r2 = jax.random.split(rng, 3)

        noise_pos = jax.random.uniform(r1, (self.mj_model.nq,), minval=-0.04, maxval=0.04)
        qpos = jnp.array(self.mj_model.qpos0)
        qpos = qpos.at[2].set(Z_TARGET)
        
        # Rumore iniziale per favorire l'esplorazione
        qpos = qpos + noise_pos * 0.02
        
        q = qpos[3:7]
        qpos = qpos.at[3:7].set(q / jnp.linalg.norm(q))

        qvel = jax.random.uniform(r2, (self.mj_model.nv,), minval=-0.01, maxval=0.01)

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state)

        zero = jnp.float32(0.0)
        metrics = {'reward': zero}
        
        return State(pipeline_state, obs, zero, zero, metrics, {'rng': rng, 'step': jnp.int32(0)})

    def step(self, state: State, action: jnp.ndarray) -> State:
        rng, r_act, r_obs = jax.random.split(state.info['rng'], 3)
        step_n = state.info['step'] + 1

        # ----------------------------------------------------------------
        # 1. CONTROLLO DIRETTO (Policy -> Giunti)
        # ----------------------------------------------------------------
        # L'azione in uscita dalla policy (solitamente una gaussiana normalizzata o tanh)
        # viene scalata nel range fisico richiesto: [-2.5, 2.5] radianti.
        ctrl = jnp.tanh(action) * 2.5
        
        # Rumore di attuazione per robustezza Sim-to-Real
        ctrl = ctrl + jax.random.normal(r_act, ctrl.shape) * 0.02
        ctrl = jnp.clip(ctrl, -2.5, 2.5)

        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)

        obs = self._get_obs(pipeline_state)
        obs = obs + jax.random.normal(r_obs, obs.shape) * 0.003

        # ----------------------------------------------------------------
        # 2. REWARD SHAPING (Semplificato)
        # ----------------------------------------------------------------
        
        # Reward puramente basato sulla velocità di avanzamento X
        ref_vel = 1.0 
        vel_y = pipeline_state.qvel[1]
        err_lin = jnp.square(vel_y-ref_vel)
        SIGMA_LIN = 0.5
        reward_y= 1.2 * jnp.exp(-err_lin / SIGMA_LIN)


        vel_x = pipeline_state.qvel[0]
        reward_x= 0.3 * jnp.abs(vel_x)

        last_action = state.info.get('last_action', jnp.zeros_like(action))
        p_smooth = 0.8 * jnp.sum(jnp.square(action - last_action))

        # Condizioni di caduta (il robot tocca a terra col telaio o si ribalta)
        torso_z = pipeline_state.qpos[2]
        tilt = jnp.square(pipeline_state.qpos[4]) + jnp.square(pipeline_state.qpos[5])
        is_fallen = (torso_z < 0.1) | (torso_z > 0.50) | (tilt > 0.3)
        p_fall = jnp.where(is_fallen, -100.0, 0.0)


        joint_angles = pipeline_state.qpos[7:19]
        p_posture = 1 * jnp.mean(jnp.square(joint_angles))
        safe_limit = 0.7 
        excess_angle = jnp.maximum(0.0, jnp.abs(joint_angles) - safe_limit)
        p_limits = 0.5 * jnp.sum(jnp.square(excess_angle))

        total_reward = (
            reward_y
            -reward_x
            + p_fall
            - p_posture
            - p_limits
            - p_smooth
        )

        new_info = state.info.copy()
        new_info.update({'last_action': action})
        # Termina l'episodio se cade o se supera MAX_STEPS
        done = jnp.where(is_fallen | (step_n >= MAX_STEPS), 1.0, 0.0)

        metrics = {'reward': total_reward}

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=total_reward,
            done=done,
            metrics=metrics,
            info={'rng': rng, 'step': step_n},
        )