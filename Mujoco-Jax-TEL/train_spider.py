"""
train_spider.py  — PPO per SpiderBot Goal-Conditioned (AMD Ryzen 7 7700)
========================================================================
Ottimizzato per JAX/MJX. Include log di debug per monitorare la compilazione JIT.
"""

import os

# Configurazione XLA per AMD Ryzen 7 7700 (8 core fisici)
os.environ['XLA_FLAGS']                  = '--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=1'
os.environ['OMP_NUM_THREADS']            = '8'
os.environ['MKL_NUM_THREADS']            = '8'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial
from spider_env import SpiderBotEnv, OBS_DIM, ACT_DIM

# ================= IPERPARAMETRI ================= #
N_ENVS          = 256     
ROLLOUT_STEPS   = 256     
TOTAL_TIMESTEPS = 20_000_000 

GAMMA           = 0.99    
GAE_LAMBDA      = 0.95    
CLIP_EPS        = 0.2     
ENT_COEF        = 0.1
VF_COEF         = 0.5     
MAX_GRAD_NORM   = 0.5     

LEARNING_RATE   = 1e-4    
N_EPOCHS        = 5       
MINIBATCH_SIZE  = 4096    

HIDDEN_SIZES    = [256, 256, 256]   
LOG_STD_INIT    = -0.5              

SAVE_EVERY_ITER = 25      
BACKUP_FILE     = "spider_policy_best.pkl"
LATEST_FILE     = "spider_policy_latest.pkl"

# ================= RETE NEURALE ================= #
def orthogonal_init(scale: float = 1.0):
    return nn.initializers.orthogonal(scale)

class ActorCritic(nn.Module):
    hidden_sizes: tuple = (256, 256, 128)

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        x = obs
        for h in self.hidden_sizes[:-1]:
            x = nn.Dense(h, kernel_init=orthogonal_init(np.sqrt(2)))(x)
            x = nn.elu(x)

        a = nn.Dense(self.hidden_sizes[-1], kernel_init=orthogonal_init(np.sqrt(2)))(x)
        a = nn.elu(a)
        mean = nn.Dense(ACT_DIM, kernel_init=orthogonal_init(0.01))(a)

        log_std = self.param('log_std', nn.initializers.constant(LOG_STD_INIT), (ACT_DIM,))

        v = nn.Dense(self.hidden_sizes[-1], kernel_init=orthogonal_init(np.sqrt(2)))(x)
        v = nn.elu(v)
        value = nn.Dense(1, kernel_init=orthogonal_init(1.0))(v)

        return mean, log_std, value.squeeze(-1)

# ================= UTILITY ================= #
def gaussian_log_prob(action, mean, log_std):
    std = jnp.exp(log_std)
    log_p = -0.5 * jnp.sum(
        jnp.square((action - mean) / (std + 1e-8)) + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1
    )
    return log_p

def gaussian_entropy(log_std):
    return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)

def sample_action(mean, log_std, rng):
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng, mean.shape)
    action = mean + std * noise  
    log_p = gaussian_log_prob(action, mean, log_std)
    return action, log_p

# ================= CORE RL ================= #
def collect_rollout(env, params, model, states, rng, rollout_steps: int):
    def step_fn(carry, _):
        states, rng = carry
        rng, rng_act = jax.random.split(rng)
        
        obs_batch = states.obs  
        means, log_stds, values = jax.vmap(lambda o: model.apply(params, o))(obs_batch)
        rngs = jax.random.split(rng_act, N_ENVS)
        actions, log_probs = jax.vmap(sample_action)(means, log_stds, rngs)

        rng, rng_step = jax.random.split(rng)
        next_states = jax.vmap(env.step)(states, actions)

        rng, rng_reset = jax.random.split(rng)
        reset_rngs = jax.random.split(rng_reset, N_ENVS)
        fresh_states = jax.vmap(env.reset)(reset_rngs)

        dones = next_states.done  
        
        def reset_if_done(fresh: jnp.ndarray, cont: jnp.ndarray) -> jnp.ndarray:
            expand_shape = (dones.shape[0],) + (1,) * (cont.ndim - 1)
            done_expanded = dones.reshape(expand_shape)
            return jnp.where(done_expanded, fresh, cont)

        states_after = jax.tree_util.tree_map(reset_if_done, fresh_states, next_states)
        
        transition = (obs_batch, actions, log_probs, next_states.reward, dones, values)
        return (states_after, rng), transition

    (states, rng), buffer = jax.lax.scan(step_fn, (states, rng), None, length=rollout_steps)
    return states, rng, buffer

@jax.jit
def compute_gae(rewards, dones, values, last_values, gamma=GAMMA, lam=GAE_LAMBDA):
    T, N = rewards.shape
    def gae_step(carry, t):
        last_gae = carry
        next_val = jnp.where(t == T - 1, last_values, values[t + 1])
        next_val = jnp.where(dones[t] > 0.5, 0.0, next_val)
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        return gae, gae

    _, advantages = jax.lax.scan(gae_step, jnp.zeros(N), jnp.arange(T - 1, -1, -1))
    advantages = jnp.flip(advantages, axis=0)
    return advantages, advantages + values

@partial(jax.jit, static_argnums=(2,))
def get_last_values(params, obs, model):
    _, _, values = jax.vmap(lambda o: model.apply(params, o))(obs)
    return values

@partial(jax.jit, static_argnums=(2,))
def ppo_update(params, opt_state, model, batch, rng, lr_schedule_val):
    obs, actions, old_log_probs, advantages, returns = batch
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    def epoch_fn(carry, _):
        params, opt_state, rng = carry
        rng, rng_perm = jax.random.split(rng)
        perm = jax.random.permutation(rng_perm, len(obs))
        obs_s, act_s, olp_s, adv_s, ret_s = obs[perm], actions[perm], old_log_probs[perm], advantages[perm], returns[perm]

        def minibatch_fn(carry, mb_idx):
            params, opt_state = carry
            start = mb_idx * MINIBATCH_SIZE
            mb_obs = jax.lax.dynamic_slice_in_dim(obs_s, start, MINIBATCH_SIZE)
            mb_act = jax.lax.dynamic_slice_in_dim(act_s, start, MINIBATCH_SIZE)
            mb_olp = jax.lax.dynamic_slice_in_dim(olp_s, start, MINIBATCH_SIZE)
            mb_adv = jax.lax.dynamic_slice_in_dim(adv_s, start, MINIBATCH_SIZE)
            mb_ret = jax.lax.dynamic_slice_in_dim(ret_s, start, MINIBATCH_SIZE)

            def loss_fn(params):
                means, log_stds, values = jax.vmap(lambda o: model.apply(params, o))(mb_obs)
                log_probs = jax.vmap(gaussian_log_prob)(mb_act, means, log_stds)
                
                ratio = jnp.exp(log_probs - mb_olp)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))
                value_loss = jnp.mean(jnp.square(values - mb_ret))
                entropy = jnp.mean(jax.vmap(gaussian_entropy)(log_stds))

                total_loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy
                return total_loss, {'pl': policy_loss, 'vl': value_loss, 'ent': entropy}

            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), metrics

        (params, opt_state), mb_metrics = jax.lax.scan(minibatch_fn, (params, opt_state), jnp.arange(len(obs) // MINIBATCH_SIZE))
        return (params, opt_state, rng), mb_metrics

    (params, opt_state, rng), all_metrics = jax.lax.scan(epoch_fn, (params, opt_state, rng), None, length=N_EPOCHS)
    return params, opt_state, jax.tree_util.tree_map(jnp.mean, all_metrics)

def train():
    print("=" * 60)
    print("  SpiderBot PPO: Goal-Conditioned Locomotion (CON DEBUG)")
    print(f"  Batch size : {N_ENVS * ROLLOUT_STEPS:,} | N_ENVS: {N_ENVS}")
    print("=" * 60)

    env = SpiderBotEnv(xml_path="spiderbot.xml")
    model = ActorCritic(hidden_sizes=tuple(HIDDEN_SIZES))
    rng = jax.random.PRNGKey(42)

    if os.path.exists(BACKUP_FILE):
        print(f"[INFO] Trovato {BACKUP_FILE}. Riprendo l'addestramento!")
        with open(BACKUP_FILE, "rb") as f: params = pickle.load(f)
    else:
        print("[INFO] Nessun salvataggio trovato. Inizializzo rete da zero.")
        rng, rng_init = jax.random.split(rng)
        params = model.init(rng_init, jnp.zeros((OBS_DIM,)))

    global optimizer
    optimizer = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LEARNING_RATE))
    opt_state = optimizer.init(params)

    rng, rng_reset = jax.random.split(rng)
    states = jax.vmap(env.reset)(jax.random.split(rng_reset, N_ENVS))

    @jax.jit
    def jit_collect(params_in, states_in, rng_in):
        return collect_rollout(env, params_in, model, states_in, rng_in, ROLLOUT_STEPS)

    best_reward = -float('inf')
    global_step, iteration = 0, 0
    B = N_ENVS * ROLLOUT_STEPS

    try:
        while True:
            t_iter = time.time()
            rng, rng_roll = jax.random.split(rng)
            
            # --------- START DEBUG LOGGING ---------
            if iteration == 0:
                print("\n[ATTENZIONE] --- PRIMA ITERAZIONE JAX ---")
                print("Il compilatore XLA sta traducendo le funzioni in linguaggio macchina per la CPU.")
                print("Il sistema NON è bloccato, ma questa fase richiederà diversi minuti.\n")

            print(f"[Iter {iteration}] 1. Avvio raccolta Rollout (256 step per 256 ambienti)...")
            t_rollout_start = time.time()
            
            # --- POTENZIALE PUNTO DI "BLOCCO" (COMPILAZIONE JIT) ---
            states, rng, buffer = jit_collect(params, states, rng_roll)
            print(f" -> Rollout completato in {time.time() - t_rollout_start:.2f} secondi.")

            obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = buffer

            print(f"[Iter {iteration}] 2. Elaborazione Vantaggi (GAE)...")
            t_gae_start = time.time()
            last_values = get_last_values(params, states.obs, model)
            advantages, returns = compute_gae(rew_buf, done_buf, val_buf, last_values)
            print(f" -> Vantaggi elaborati in {time.time() - t_gae_start:.2f} secondi.")

            batch = (obs_buf.reshape(B, OBS_DIM), act_buf.reshape(B, ACT_DIM), 
                     logp_buf.reshape(B), advantages.reshape(B), returns.reshape(B))

            print(f"[Iter {iteration}] 3. Avvio Ottimizzazione Rete Neurale (PPO Update)...")
            t_update_start = time.time()
            rng, rng_upd = jax.random.split(rng)
            
            # --- SECONDO POTENZIALE PUNTO DI "BLOCCO" (COMPILAZIONE JIT) ---
            params, opt_state, metrics = ppo_update(params, opt_state, model, batch, rng_upd, LEARNING_RATE)
            print(f" -> Update PPO completato in {time.time() - t_update_start:.2f} secondi.\n")
            # --------- END DEBUG LOGGING ---------

            global_step += B
            mean_reward = float(jnp.mean(rew_buf))
            fps = B / (time.time() - t_iter)

            # Stampa le metriche principali sempre per le prime 5 iterazioni, poi ogni 5
            if iteration < 5 or iteration % 5 == 0:
                print(f">>> SOMMARIO ITERAZIONE {iteration:4d} | Rew/step: {mean_reward:+7.3f} | Policy Loss: {float(metrics['pl']):+6.3f} | Velocità: {fps:.0f} FPS <<<\n")

            if mean_reward > best_reward:
                best_reward = mean_reward
                with open(BACKUP_FILE, "wb") as f: pickle.dump(params, f)

            iteration += 1

    except KeyboardInterrupt:
        print("\n[STOP] Salvataggio finale in corso...")
        with open("spider_policy_final.pkl", "wb") as f: pickle.dump(params, f)
        print("[OK] Completato!")

if __name__ == "__main__":
    train()