cat << 'EOF' > README.md
# 🕷️ SpiderBot

> **A 12-DOF Quadrupedal Robot**
> *Daniele Palomba & Giulio Vestri* — Progetto Finale di Robotica Avanzata, Università degli Studi di Napoli Federico II.

---

## 🚀 Panoramica del Progetto
**SpiderBot** è un robot quadrupede biomimetico a **12 gradi di libertà (DOF)** ingegnerizzato combinando un'estetica street-design ad un'architettura di controllo robusta e scalabile. Il progetto copre l'intero stack meccatronico: dalla progettazione geometrica esagonale alla simulazione e all'addestramento tramite **Deep Reinforcement Learning (DRL)** end-to-end, fino all'implementazione fisica su hardware reale.

### ✨ Caratteristiche Principali
* **Geometria Esagonale Ottimizzata:** Telaio a simmetria esagonale regolare per garantire una base di supporto statico eccezionalmente ampia e stabile.
* **Doppio Paradigma di Controllo:**
  * *Cinematica Open-Loop:* Generatore di andature analitiche tempo-varianti (Creep, Trot, Gallop, Pace).
  * *End-to-End Neural Controller:* Controllo basato su policy DRL apprese direttamente in simulazione fisica accelerata.
* **Simulazione ad Altissime Prestazioni:** Pipeline basata su **JAX, Brax e MuJoCo MJX** per un addestramento massivamente parallelo su GPU/CPU.
* **Hardware Efficiente:** Architettura Master-Slave basata su MCU Arduino e driver PWM PCA9685 via I2C per la gestione sincrona di 12 servomotori MZ996R.

---

## 🛠️ Architettura di Controllo

### 1. Controllo Cinematico Open-Loop
La pipeline cinematica classica implementata in ambiente MATLAB/Simulink è strutturata su tre livelli gerarchici:
1. **High-level Trajectory Planner:** Calcola le coordinate cartesiane 3D desiderate di ogni piede rispetto al frame solidale al robot.
2. **Mid-level Spatial Transformation Layer:** Converte i footprint cartesiani in angoli di giunto tramite un solutore geometrico di Cinematica Inversa (IK).
3. **Low-level Actuation Layer:** Invia i comandi PWM ai singoli servomotori (Coxa, Femur, Tibia) sincronizzando il digital twin.

#### 🐾 Andature Implementate
| Andatura | Tipo | Parametri di Foot Scheduling | Descrizione |
| :--- | :--- | :--- | :--- |
| **Creep Gait** | 4 battute | `[0, 25, 50, 75]` (T = 2.0s) | Massima stabilità su terreni sconnessi. |
| **Trot Gait** | 2 battute | `[0, 50, 50, 0]` (T = 0.6s) | Andatura a velocità medio-alta con coppie diagonali sincronizzate. |
| **Gallop Gait** | Alta velocità | Sincronizzazione ad impulsi | Massima reattività dinamica. |
| **Pace Gait** | 2 battute | Sincronizzazione laterale | Movimento coordinato degli arti dello stesso lato. |

### 2. Deep Reinforcement Learning (End-to-End)
In alternativa al controllo classico, SpiderBot implementa un paradigma **Neural Controller End-to-End**. La rete neurale mappa direttamente lo stato fisico in target angolari per i giunti.

* **Observation Space (35D):** Include orientamento (quaternioni), velocità angolare, andamento e posizione corrente dei giunti.
* **Action Space (12D):** Target angolari $[-2.5, 2.5]$ rad per i 12 motori.
* **Teleoperated Learning Reward:** Ottimizzazione del tracking multi-assiale tramite funzioni di ricompensa gaussiane esponenziali ed eliminazione dei fenomeni di drift laterale ($v_x$) tramite penalità dedicate.

---

## 💻 Struttura della Repository
```bash
├── SpiderBot/             # Firmware Arduino per il controllo locale (IK math & PWM execution)
├── MatLab/                # Modelli Simulink open-loop e script di plotting cinematica
│   ├── STL/               # Modelli tridimensionali del telaio e dei link
│   └── open_loop.slx      # Modello di simulazione principale
├── MuJoCo-Jax/            # Pipeline di Reinforcement Learning massivamente parallelo
│   ├── spider_env.py      # Definizione dell'ambiente Brax / MuJoCo MJX
│   ├── train_spider.py    # Script di addestramento PPO
│   └── spiderbot.xml      # Descrizione MJCF del robot
└── Mujoco-Jax-TEL/        # Script di telemetria e testing delle policy caricate
''

---

## 🔌 Architettura Elettronica e Hardware' 
Il sistema fisico si basa su una topologia **Master-Slave** per ottimizzare i cicli di calcolo:

* **Host Esterno (Master):** Gestisce lo streaming di comandi ad alto livello o l'\''esecuzione della policy neurale.' \
* **MCU Locale (Slave - Arduino):** Riceve i dati seriali, calcola le fasi dell'\''andatura ed esegue l'\''algebra geometrica della Cinematica Inversa in tempo reale.
* **PCA9685 PWM Driver:** Attuatore I2C dedicato per scaricare i timer della MCU e pilotare con impulsi precisi a 12-bit i **12 servomotori MZ996R**.
* **Chassis:** Interamente stampato in 3D tramite tecnologia FDM (PLA), rispettando al millimetro le dimensioni cinematiche del modello URDF.' 
'---
'' \
'## 🚀 Come Iniziare' \
'' \
'### 1. Simulazione Cinematica (MATLAB)' \
'1. Apri MATLAB ed accedi alla cartella `MatLab/`.' \
'2. Esegui `graphs.m` per inizializzare le variabili d'\''andatura.' \
'3. Apri e lancia il modello Simulink `open_loop.slx`.' \
'' \
'### 2. Training DRL (Python/JAX)' \
'Assicurati di disporre di un ambiente con supporto CUDA per sfruttare MJX:' \
'```bash' \
'cd MuJoCo-Jax' \
'pip install -r requirements.txt' \
'python train_spider.py' \
'```' \
'Per visualizzare i risultati o fare il deploy di una policy pre-addestrata:' \
'```bash' \
'python enjoy_spider.py --policy spider_policy_best.pkl' \
'```' \
'' \
'---' \
'*Progetto sviluppato in collaborazione con il **PRISMA Lab**.*' >> README.md