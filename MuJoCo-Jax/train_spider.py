"""
train_spider.py  — PPO per SpiderBot su CPU AMD Ryzen 7 7700
=============================================================
Architettura:
  - PPO (Proximal Policy Optimization) con GAE
  - N_ENVS ambienti paralleli via jax.vmap
  - Rete Actor-Critic 256->256->128 con ELU e init ortogonale
  - JIT abilitato (XLA multi-thread su 8 core fisici)
  - Rollout buffer completo prima di ogni update
  - Logging dettagliato con metriche separate

Ryzen 7 7700: 8 core fisici / 16 logici / L3 32 MB
  -> N_ENVS=16, ROLLOUT_STEPS=512, MINIBATCH_SIZE=512
  -> stimato ~8-12 ore per 1000 iterazioni PPO (~8M step totali)
"""

import os

# ------------------------------------------------------------------ #
#  Configurazione XLA per Ryzen 7 7700                                #
#  8 core fisici -> 8 thread Eigen; non saturare con i logici         #
# ------------------------------------------------------------------ #
os.environ['XLA_FLAGS']                  = '--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=1'
os.environ['OMP_NUM_THREADS']            = '8'
os.environ['MKL_NUM_THREADS']            = '8'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'
# JAX_DISABLE_JIT rimosso: il JIT è fondamentale per la velocità

import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial
from spider_env import SpiderBotEnv, OBS_DIM, ACT_DIM

# ================================================================== #
#  IPERPARAMETRI                                                      #
# ================================================================== #

# Parallelismo: bilanciato per L3 32 MB e 8 core
N_ENVS          = 16      # ambienti in parallelo (vmap)
ROLLOUT_STEPS   = 512     # step per env prima di ogni update
TOTAL_TIMESTEPS = 10_000_000  # timestep totali

# PPO
GAMMA           = 0.99    # discount factor
GAE_LAMBDA      = 0.95    # GAE lambda
CLIP_EPS        = 0.2     # clip ratio PPO
ENT_COEF        = 0.0002    # coefficiente entropia (esplorazione)
VF_COEF         = 0.5     # coefficiente value loss
MAX_GRAD_NORM   = 0.5     # gradient clipping

# Ottimizzatore
LEARNING_RATE   = 5e-5    # Adam lr iniziale (con annealing)
N_EPOCHS        = 10       # epoche per update PPO
MINIBATCH_SIZE  = 512     # dimensione minibatch

# Rete
HIDDEN_SIZES    = [256, 256, 256]   # layer nascosti
LOG_STD_INIT    = -0.5              # log std deviazione iniziale policy

# Salvataggi
SAVE_EVERY_ITER = 25      # salva checkpoint ogni N iterazioni PPO
BACKUP_FILE     = "spider_policy_best.pkl"
LATEST_FILE     = "spider_policy_latest.pkl"

# ================================================================== #
#  RETE ACTOR-CRITIC                                                  #
# ================================================================== #

def orthogonal_init(scale: float = 1.0):
    """Init ortogonale: migliore convergenza per policy RL."""
    return nn.initializers.orthogonal(scale)


class ActorCritic(nn.Module):
    """
    Rete separata per actor e critic (trunk condiviso opzionale).
    Actor: output = media azioni (tanh bounded) + log_std apprendibile
    Critic: output = stima del valore V(s)
    """
    hidden_sizes: tuple = (256, 256, 128)

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        # Trunk condiviso
        x = obs
        for i, h in enumerate(self.hidden_sizes[:-1]):
            x = nn.Dense(h, kernel_init=orthogonal_init(np.sqrt(2)),
                         bias_init=nn.initializers.zeros)(x)
            x = nn.elu(x)

        # Branch actor
        a = nn.Dense(self.hidden_sizes[-1],
                     kernel_init=orthogonal_init(np.sqrt(2)),
                     bias_init=nn.initializers.zeros)(x)
        a = nn.elu(a)
        mean = nn.Dense(ACT_DIM,
                        kernel_init=orthogonal_init(0.01),
                        bias_init=nn.initializers.zeros)(a)

        # log_std come parametro apprendibile (non dipende dallo stato)
        log_std = self.param('log_std', nn.initializers.constant(LOG_STD_INIT),
                             (ACT_DIM,))

        # Branch critic
        v = nn.Dense(self.hidden_sizes[-1],
                     kernel_init=orthogonal_init(np.sqrt(2)),
                     bias_init=nn.initializers.zeros)(x)
        v = nn.elu(v)
        value = nn.Dense(1,
                         kernel_init=orthogonal_init(1.0),
                         bias_init=nn.initializers.zeros)(v)

        return mean, log_std, value.squeeze(-1)


# ================================================================== #
#  FUNZIONI PROBABILISTICHE                                           #
# ================================================================== #

def gaussian_log_prob(action, mean, log_std):
    """Log-probabilità gaussiana per azione continua."""
    std = jnp.exp(log_std)
    log_p = -0.5 * jnp.sum(
        jnp.square((action - mean) / (std + 1e-8))
        + 2 * log_std
        + jnp.log(2 * jnp.pi),
        axis=-1
    )
    return log_p


def gaussian_entropy(log_std):
    """Entropia di una gaussiana multivariata diagonale."""
    return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)

def sample_action(mean, log_std, rng):
    """Campiona azione (pura) e calcola log-prob gaussiana."""
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng, mean.shape)
    action = mean + std * noise  # Azione illimitata (Gaussiana)
    
    # Nessun Tanh squashing qui! Teniamo l'azione pura per il buffer
    log_p = gaussian_log_prob(action, mean, log_std)
    
    return action, log_p


# ================================================================== #
#  ROLLOUT: raccolta esperienza su N ambienti in parallelo           #
# ================================================================== #

def collect_rollout(env, params, model, states, rng, rollout_steps: int):
    """
    Raccoglie rollout_steps transizioni per N_ENVS ambienti.
    Usa lax.scan per efficienza JAX (non un loop Python).
    Returns: buffer con (obs, actions, log_probs, rewards, dones, values)
    """

    def step_fn(carry, _):
        states, rng = carry
        rng, rng_act = jax.random.split(rng)

        # Forward pass su tutti gli env in parallelo
        obs_batch = states.obs  # (N_ENVS, OBS_DIM)

        # vmap del modello su batch
        means, log_stds, values = jax.vmap(
            lambda o: model.apply(params, o)
        )(obs_batch)

        # Campiona azione per ogni env
        rngs = jax.random.split(rng_act, N_ENVS)
        actions, log_probs = jax.vmap(sample_action)(means, log_stds, rngs)

        # Step su tutti gli env
        rng, rng_step = jax.random.split(rng)
        next_states = jax.vmap(env.step)(states, actions)

        # Reset automatico degli env terminati
        rng, rng_reset = jax.random.split(rng)
        reset_rngs = jax.random.split(rng_reset, N_ENVS)
        fresh_states = jax.vmap(env.reset)(reset_rngs)

        dones = next_states.done  # (N_ENVS,)

        # -------------------------------------------------------------
        # FIX: Broadcasting dinamico N-Dimensionale per MuJoCo MJX
        # -------------------------------------------------------------
        def reset_if_done(fresh: jnp.ndarray, cont: jnp.ndarray) -> jnp.ndarray:
            # Crea una tupla di shape per espandere 'dones'. 
            # Es: Se cont è (16, 14, 6), expand_shape sarà (16, 1, 1)
            expand_shape = (dones.shape[0],) + (1,) * (cont.ndim - 1)
            done_expanded = dones.reshape(expand_shape)
            return jnp.where(done_expanded, fresh, cont)

        # Sostituisci gli env terminati con stati freschi per tutte le foglie del PyTree
        states_after = jax.tree_util.tree_map(
            reset_if_done,
            fresh_states, next_states
        )
        # -------------------------------------------------------------

        transition = (
            obs_batch,            # (N_ENVS, OBS_DIM)
            actions,              # (N_ENVS, ACT_DIM)
            log_probs,            # (N_ENVS,)
            next_states.reward,   # (N_ENVS,)
            dones,                # (N_ENVS,)
            values,               # (N_ENVS,)
        )
        return (states_after, rng), transition

    (states, rng), buffer = jax.lax.scan(
        step_fn,
        (states, rng),
        None,
        length=rollout_steps
    )

    # buffer shape: (rollout_steps, N_ENVS, ...)
    return states, rng, buffer


# ================================================================== #
#  GAE: Generalized Advantage Estimation                              #
# ================================================================== #

def compute_gae(rewards, dones, values, last_values, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    Calcola advantages con GAE e returns target per il critic.
    Input shape: (T, N) dove T=rollout_steps, N=N_ENVS
    """
    T, N = rewards.shape

    advantages = jnp.zeros((T, N))
    last_gae = jnp.zeros(N)

    # Scan al contrario
    def gae_step(carry, t):
        last_gae = carry
        # Valore del prossimo step (0 se terminato)
        next_val = jnp.where(t == T - 1, last_values, values[t + 1])
        next_val = jnp.where(dones[t] > 0.5, 0.0, next_val)

        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        return gae, gae

    _, advantages = jax.lax.scan(
        gae_step,
        last_gae,
        jnp.arange(T - 1, -1, -1)
    )
    advantages = jnp.flip(advantages, axis=0)
    returns = advantages + values
    return advantages, returns


# ================================================================== #
#  PPO UPDATE                                                         #
# ================================================================== #

@partial(jax.jit, static_argnums=(2,))
def ppo_update(params, opt_state, model, batch, rng, lr_schedule_val):
    """
    Esegue N_EPOCHS passate PPO sul buffer, con minibatch shuffle.
    """
    obs, actions, old_log_probs, advantages, returns = batch

    # Normalizza advantages (fondamentale per stabilità)
    adv_mean = jnp.mean(advantages)
    adv_std  = jnp.std(advantages) + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    def epoch_fn(carry, _):
        params, opt_state, rng = carry
        rng, rng_perm = jax.random.split(rng)

        # Shuffle
        perm = jax.random.permutation(rng_perm, len(obs))
        obs_s    = obs[perm]
        act_s    = actions[perm]
        olp_s    = old_log_probs[perm]
        adv_s    = advantages[perm]
        ret_s    = returns[perm]

        # Minibatch update
        def minibatch_fn(carry, mb_idx):
            params, opt_state = carry
            start = mb_idx * MINIBATCH_SIZE
            mb_obs  = jax.lax.dynamic_slice_in_dim(obs_s,  start, MINIBATCH_SIZE)
            mb_act  = jax.lax.dynamic_slice_in_dim(act_s,  start, MINIBATCH_SIZE)
            mb_olp  = jax.lax.dynamic_slice_in_dim(olp_s,  start, MINIBATCH_SIZE)
            mb_adv  = jax.lax.dynamic_slice_in_dim(adv_s,  start, MINIBATCH_SIZE)
            mb_ret  = jax.lax.dynamic_slice_in_dim(ret_s,  start, MINIBATCH_SIZE)

            def loss_fn(params):
                means, log_stds, values = jax.vmap(
                    lambda o: model.apply(params, o)
                )(mb_obs)

                # Log prob puro delle azioni nel buffer
                log_probs = jax.vmap(gaussian_log_prob)(mb_act, means, log_stds)
                
                # RIMOSSO: log_probs -= jnp.sum(jnp.log(1 - mb_act ** 2 + 1e-6), axis=-1)

                # PPO policy loss (resto del codice invariato)
                ratio = jnp.exp(log_probs - mb_olp)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

                # Value loss (clipped)
                value_loss = jnp.mean(jnp.square(values - mb_ret))

                # Entropia (bonus esplorazione)
                entropy = jnp.mean(jax.vmap(gaussian_entropy)(log_stds))

                total_loss = (
                    policy_loss
                    + VF_COEF * value_loss
                    - ENT_COEF * entropy
                )

                metrics = {
                    'policy_loss': policy_loss,
                    'value_loss':  value_loss,
                    'entropy':     entropy,
                    'approx_kl':   jnp.mean((ratio - 1) - jnp.log(ratio + 1e-8)),
                }
                return total_loss, metrics

            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), metrics

        n_minibatches = len(obs) // MINIBATCH_SIZE
        (params, opt_state), mb_metrics = jax.lax.scan(
            minibatch_fn,
            (params, opt_state),
            jnp.arange(n_minibatches)
        )
        return (params, opt_state, rng), mb_metrics

    (params, opt_state, rng), all_metrics = jax.lax.scan(
        epoch_fn,
        (params, opt_state, rng),
        None,
        length=N_EPOCHS
    )

    avg_metrics = jax.tree_util.tree_map(jnp.mean, all_metrics)
    return params, opt_state, avg_metrics


# ================================================================== #
#  TRAINING LOOP PRINCIPALE                                           #
# ================================================================== #

def train():
    print("=" * 60)
    print("  SpiderBot PPO Training — Ryzen 7 7700 ottimizzato")
    print("=" * 60)
    print(f"  Ambienti paralleli : {N_ENVS}")
    print(f"  Step per rollout   : {ROLLOUT_STEPS}")
    print(f"  Timestep totali    : {TOTAL_TIMESTEPS:,}")
    print(f"  Batch size         : {N_ENVS * ROLLOUT_STEPS:,}")
    print(f"  Minibatch size     : {MINIBATCH_SIZE}")
    print(f"  Epoche PPO         : {N_EPOCHS}")
    print(f"  Rete               : {HIDDEN_SIZES}")
    print("=" * 60)

    # ---- Ambiente ----
    env = SpiderBotEnv(xml_path="spiderbot.xml")

# ---- Modello ----
    model = ActorCritic(hidden_sizes=tuple(HIDDEN_SIZES))
    rng   = jax.random.PRNGKey(42)

    # ---------------------------------------------------------
    # RESUME LOGIC: Riprende l'addestramento se esiste il file
    # ---------------------------------------------------------
    if os.path.exists("spider_policy_best.pkl"):
        print(f"\n[INFO] Trovato {os.path.basename("spider_policy_best.pkl")}. Riprendo l'addestramento dai dati esistenti!")
        with open("spider_policy_best.pkl", "rb") as f:
            params = pickle.load(f)
    else:
        print("\n[INFO] Nessun salvataggio precedente. Inizializzo rete da zero.")
        rng, rng_init = jax.random.split(rng)
        params = model.init(rng_init, jnp.zeros((OBS_DIM,)))

    # ---- Ottimizzatore (Learning Rate FISSO per loop infinito) ----
    global optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(LEARNING_RATE, eps=1e-5)  # Nessun decadimento lineare
    )
    opt_state = optimizer.init(params)

    # ---- Reset iniziale di N_ENVS ambienti ----
    rng, rng_reset = jax.random.split(rng)
    reset_rngs = jax.random.split(rng_reset, N_ENVS)
    states = jax.vmap(env.reset)(reset_rngs)

    # ---- Precompila collect_rollout ----
    print("\n[INFO] Compilazione JIT (prima iterazione più lenta)...")
    @jax.jit
    def jit_collect(params_in, states_in, rng_in):
        return collect_rollout(env=env, params=params_in, model=model, states=states_in, rng=rng_in, rollout_steps=ROLLOUT_STEPS)

    best_reward = -float('inf')
    global_step = 0
    t_start = time.time()
    iteration = 0

    print("[INFO] Addestramento iniziato. Premi [Ctrl+C] per interrompere e salvare.\n")

    # ---------------------------------------------------------
    # LOOP INFINITO CON SALVATAGGIO SICURO (Ctrl+C)
    # ---------------------------------------------------------
    try:
        while True:
            t_iter = time.time()

            # [IL CODICE INTERNO RIMANE IDENTICO A PRIMA]
            rng, rng_roll = jax.random.split(rng)
            states, rng, buffer = jit_collect(params, states, rng_roll)

            obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = buffer

            last_means, last_log_stds, last_values = jax.vmap(
                lambda o: model.apply(params, o)
            )(states.obs)

            advantages, returns = compute_gae(rew_buf, done_buf, val_buf, last_values)

            B = N_ENVS * ROLLOUT_STEPS
            obs_flat    = obs_buf.reshape(B, OBS_DIM)
            act_flat    = act_buf.reshape(B, ACT_DIM)
            logp_flat   = logp_buf.reshape(B)
            adv_flat    = advantages.reshape(B)
            ret_flat    = returns.reshape(B)

            batch = (obs_flat, act_flat, logp_flat, adv_flat, ret_flat)

            rng, rng_upd = jax.random.split(rng)
            params, opt_state, metrics = ppo_update(
                params, opt_state, model, batch, rng_upd, LEARNING_RATE
            )

            global_step += B

           # ---- Logging all'interno di train_spider.py ----
            mean_reward = float(jnp.mean(rew_buf))
            ep_reward   = float(jnp.mean(jnp.sum(rew_buf, axis=0)))
            iter_time   = time.time() - t_iter
            fps         = B / iter_time

            if iteration % 5 == 0:
                print(
                    f"Iter {iteration:4d} | Rew/step: {mean_reward:+7.3f} | Ep.Rew: {ep_reward:+8.1f} | "
                    f"PL: {float(metrics['policy_loss']):+6.3f} | VL: {float(metrics['value_loss']):5.3f} | "
                    f"FPS: {fps:.0f} | Step: {global_step//1000}k"
                )

            # ---- Salvataggi ----
            if mean_reward > best_reward:
                best_reward = mean_reward
                _save(params, BACKUP_FILE)
                if iteration % 5 == 0:
                    print(f"  [NEW BEST] reward={best_reward:.4f} -> {BACKUP_FILE}")

            if iteration % SAVE_EVERY_ITER == 0:
                _save(params, LATEST_FILE)

            iteration += 1

    except KeyboardInterrupt:
        print("\n[STOP] Addestramento interrotto manualmente dall'utente!")

    # Questo blocco viene eseguito SEMPRE alla fine, sia in caso di interruzione che di crash controllato
    _save(params, "spider_policy_final.pkl")
    print(f"[OK] Salvataggio finale completato! Puoi estrarre i pesi o riavviare lo script per continuare.")


def _save(params, path):
    with open(path, "wb") as f:
        pickle.dump(params, f)


if __name__ == "__main__":
    train()