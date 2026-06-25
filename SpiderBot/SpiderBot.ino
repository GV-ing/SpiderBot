#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define SERVO_MIN 120   
#define SERVO_MAX 520   
#define SERVO_MID 320   

const int SERVO_PINS[4][3] = {
  {14, 13, 12},  // FR (Gamba 4: Indice 3)
  {10, 9, 8},   // FL (Gamba 3: Indice 2)
  {2, 1, 0},    // RL (Gamba 1: Indice 0)
  {6, 5, 4}    // RR (Gamba 2: Indice 1)
};


const float L_COXA  = 38.0;
const float L_FEMUR = 86.25;
const float L_TIBIA = 160.5;
const float R_HEX   = 125.0;

const float THETA[4] = {2.0*PI/3.0, -2.0*PI/3.0, PI/3.0, -PI/3.0};
float SPALLE_X[4];
float SPALLE_Y[4];

const float STEP_LENGTH = 50.0;
const float STEP_HEIGHT = 40.0;
const float HOME_X = 80.0, HOME_Y = 0.0, HOME_Z = -130.0;

// Tempi differenti per le due andature
const float CREEP_DUR_MS = 1000.0; 
const float TROT_DUR_MS  = 500.0;  // Il trotto deve essere più rapido per mantenere l'inerzia!

// =========================================================
// NUOVO PACCHETTO SERIALE (16 Bytes)
// =========================================================
union CommandBuffer {
  byte bytes[13]; // 12 per i float + 1 per il modo
  struct {
    float vx;
    float vy;
    float wz;
    byte gait_mode; // 0 = Creep, 1 = Trot
  } val;
} cmd;

float v_cmd_x = 0.0;
float v_cmd_y = 0.0;
float v_cmd_z = 0.0;
byte current_gait = 0; 

// =========================================================
// SETUP
// =========================================================
void setup() {
  Serial.begin(115200);
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(50);
  
  for(int i=0; i<4; i++) {
    SPALLE_X[i] = R_HEX * cos(THETA[i]);
    SPALLE_Y[i] = R_HEX * sin(THETA[i]);
  }
  delay(1000);
}

void loop() {
  leggiSeriale(); 
  unsigned long t_ms = millis();
  
  for(int i = 0; i < 4; i++) {
    float Px, Py, Pz;
    float q1, q2, q3;
    
    // Switch delle Andature: 0=Creep, 1=Trot
    if (current_gait == 0) {
      computeCreep(t_ms, i, Px, Py, Pz);
    } else {
      computeTrot(t_ms, i, Px, Py, Pz);
    }
    
    computeIK(i, Px, Py, Pz, q1, q2, q3);
    setServoRad(SERVO_PINS[i][0], q1);
    setServoRad(SERVO_PINS[i][1], q2);
    setServoRad(SERVO_PINS[i][2], q3);
  }
  delay(10); 
}

// =========================================================
// 1. CREEP GAIT 
// =========================================================
void computeCreep(unsigned long t_ms, int leg_idx, float &Px, float &Py, float &Pz) {
  float norm_v = sqrt(v_cmd_x*v_cmd_x + v_cmd_y*v_cmd_y);
  float dir_x = v_cmd_x, dir_y = v_cmd_y;
  if (norm_v > 0.001) { dir_x /= norm_v; dir_y /= norm_v; }

  float dyn_step_height = STEP_HEIGHT;
  float stance_depth = -3.0; 
  float max_rot_rad = 15.0 * PI / 180.0;
  
  float t = (float)t_ms / CREEP_DUR_MS;
  float phase_globale = t - floor(t);
  
  float offsets[4] = {0.50, 0.00, 0.75, 0.25}; 
  float local_phase = phase_globale + offsets[leg_idx];
  if (local_phase >= 1.0) local_phase -= 1.0;
  
  float th = THETA[leg_idx];
  float c_th = cos(th), s_th = sin(th);
  float x_home_g = SPALLE_X[leg_idx] + (HOME_X*c_th - HOME_Y*s_th);
  float y_home_g = SPALLE_Y[leg_idx] + (HOME_X*s_th + HOME_Y*c_th);
  
  if (norm_v < 0.001 && abs(v_cmd_z) < 0.001) {
      Px = x_home_g; Py = y_home_g; Pz = HOME_Z; return;
  }

  float step_x = 0, step_y = 0, step_z = 0, step_rot = 0;
  if (local_phase < 0.25) { 
    float swingPhase = local_phase * 4.0;
    step_x = dir_x * ((-STEP_LENGTH/2.0) + (STEP_LENGTH * swingPhase));
    step_y = dir_y * ((-STEP_LENGTH/2.0) + (STEP_LENGTH * swingPhase));
    step_rot = v_cmd_z * ((-max_rot_rad/2.0) + (max_rot_rad * swingPhase));
    step_z = sin(swingPhase * PI) * dyn_step_height;
  } else { 
    float stancePhase = (local_phase - 0.25) / 0.75;
    step_x = dir_x * ((STEP_LENGTH/2.0) - (STEP_LENGTH * stancePhase));
    step_y = dir_y * ((STEP_LENGTH/2.0) - (STEP_LENGTH * stancePhase));
    step_rot = v_cmd_z * ((max_rot_rad/2.0) - (max_rot_rad * stancePhase));
    step_z = stance_depth;
  }

  float c_rot = cos(step_rot), s_rot = sin(step_rot);
  Px = (x_home_g * c_rot - y_home_g * s_rot) + step_x;
  Py = (x_home_g * s_rot + y_home_g * c_rot) + step_y;
  Pz = HOME_Z + step_z;
}

// =========================================================
// 2. TROT GAIT (Duty Factor 50%, Offset Diagonali)
// =========================================================
void computeTrot(unsigned long t_ms, int leg_idx, float &Px, float &Py, float &Pz) {
  float norm_v = sqrt(v_cmd_x*v_cmd_x + v_cmd_y*v_cmd_y);
  float dir_x = v_cmd_x, dir_y = v_cmd_y;
  if (norm_v > 0.001) { dir_x /= norm_v; dir_y /= norm_v; }

  float dyn_step_height = STEP_HEIGHT;
  if (abs(dir_y) > 0.5) dyn_step_height *= 1.5;
  float stance_depth = 7.0; 
  float max_rot_rad = 15.0 * PI / 180.0;
  
  float t = (float)t_ms / TROT_DUR_MS;
  float phase_globale = t - floor(t);
  
  // Offsets Trotto: [RL, RR, FL, FR]
  // FL e RR sincroni, FR e RL sincroni a sfasamento mezzo ciclo
  float offsets[4] = {0.50, 0.00, 0.00, 0.50}; 
  float local_phase = phase_globale + offsets[leg_idx];
  if (local_phase >= 1.0) local_phase -= 1.0;
  
  float th = THETA[leg_idx];
  float c_th = cos(th), s_th = sin(th);
  float x_home_g = SPALLE_X[leg_idx] + (HOME_X*c_th - HOME_Y*s_th);
  float y_home_g = SPALLE_Y[leg_idx] + (HOME_X*s_th + HOME_Y*c_th);
  
  if (norm_v < 0.001 && abs(v_cmd_z) < 0.001) {
      Px = x_home_g; Py = y_home_g; Pz = HOME_Z; return;
  }

  float step_x = 0, step_y = 0, step_z = 0, step_rot = 0;
  
  // Duty Factor 50%
  if (local_phase < 0.50) { // Fase di VOLO (0.0 -> 0.5)
    float swingPhase = local_phase * 2.0; 
    step_x = dir_x * ((-STEP_LENGTH/2.0) + (STEP_LENGTH * swingPhase));
    step_y = dir_y * ((-STEP_LENGTH/2.0) + (STEP_LENGTH * swingPhase));
    step_rot = v_cmd_z * ((-max_rot_rad/2.0) + (max_rot_rad * swingPhase));
    step_z = sin(swingPhase * PI) * dyn_step_height;
  } else { // Fase di APPOGGIO (0.5 -> 1.0)
    float stancePhase = (local_phase - 0.50) * 2.0; 
    step_x = dir_x * ((STEP_LENGTH/2.0) - (STEP_LENGTH * stancePhase));
    step_y = dir_y * ((STEP_LENGTH/2.0) - (STEP_LENGTH * stancePhase));
    step_rot = v_cmd_z * ((max_rot_rad/2.0) - (max_rot_rad * stancePhase));
    step_z = stance_depth;
  }

  float c_rot = cos(step_rot), s_rot = sin(step_rot);
  Px = (x_home_g * c_rot - y_home_g * s_rot) + step_x;
  Py = (x_home_g * s_rot + y_home_g * c_rot) + step_y;
  Pz = HOME_Z + step_z;
}

// =========================================================
// CINEMATICA E SERIALE
// =========================================================
void computeIK(int leg_idx, float Px, float Py, float Pz, float &q1, float &q2, float &q3) {
  float dx = Px - SPALLE_X[leg_idx], dy = Py - SPALLE_Y[leg_idx], dz = Pz;
  float c = cos(-THETA[leg_idx]), s = sin(-THETA[leg_idx]);
  
  float loc_x = dx * c - dy * s, loc_y = dx * s + dy * c, loc_z = dz;
  q1 = atan2(loc_y, loc_x);
  
  float L_forward = sqrt(loc_x*loc_x + loc_y*loc_y) - L_COXA;
  float D = constrain(sqrt(L_forward*L_forward + loc_z*loc_z), abs(L_FEMUR - L_TIBIA)+0.001, L_FEMUR + L_TIBIA - 0.001);
  
  q3 = acos(constrain((D*D - L_FEMUR*L_FEMUR - L_TIBIA*L_TIBIA)/(2.0*L_FEMUR*L_TIBIA), -1.0, 1.0));
  q2 = atan2(loc_z, L_forward) + atan2(L_TIBIA * sin(q3), L_FEMUR + L_TIBIA * cos(q3));
  
}

void setServoRad(int channel, float angle_rad) {
  float angle_deg = constrain((angle_rad * 180.0 / PI) + 90.0, 0.0, 180.0);
  pwm.setPWM(channel, 0, map(angle_deg, 0, 180, SERVO_MIN, SERVO_MAX));
}

// Ora legge 16 Bytes (2 header + 13 payload + 1 checksum)
void leggiSeriale() {
  while (Serial.available() >= 16) { 
    if (Serial.read() == 0xAA && Serial.read() == 0x55) { 
      byte buffer[14];
      Serial.readBytes(buffer, 14); 
      
      byte calc_chk = 0;
      for(int i=0; i<13; i++) {
        calc_chk ^= buffer[i];
        cmd.bytes[i] = buffer[i];
      }
      
      if(calc_chk == buffer[13]) { 
        v_cmd_x = cmd.val.vx;
        v_cmd_y = cmd.val.vy;
        v_cmd_z = cmd.val.wz;
        current_gait = cmd.val.gait_mode; // Aggiornamento istantaneo dell'andatura!
      }
    }
  }
}