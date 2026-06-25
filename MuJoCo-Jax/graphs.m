%% Script Principale: Generazione Grafici con stile LaTeX
close all; clc;

% 1. Caricamento dati
dati = load('sim_data.mat');
t = dati.time';

% 2. Organizzazione dei dati in matrici [N_samples x 3]
% Estrazione Posizione (XYZ)
pos_data = [dati.pos(:, 1), dati.pos(:, 2), dati.pos(:, 3)];

% Estrazione Velocità (XYZ)
vel_data = [dati.vel(:, 1), dati.vel(:, 2), dati.vel(:, 3)];

% Estrazione Rotazione (Conversione Quaternioni -> RPY)
% Assumendo che i quaternioni siano nelle colonne 4-7 del file (w,x,y,z)
quat = dati.pos(:, 4:7);
eul = quat2eul(quat, 'ZYX'); % Restituisce [Yaw, Pitch, Roll]
rpy_data = [eul(:,3), eul(:,2), eul(:,1)]; % Convertiamo in [Roll, Pitch, Yaw]

%% 3. Generazione Grafici tramite la funzione personalizzata

% --- Plot Position ---
personal_plot_multi(t, pos_data, ...
    'Time [s]', 'Position [m]', ...
    {'x', 'y', 'z'}, 'north', 'Position_Creep.pdf');

% --- Plot Velocity ---
personal_plot_multi(t, vel_data, ...
    'Time [s]', 'Velocity [m/s]', ...
    {'v_x', 'v_y', 'v_z'}, 'north', 'Velocity_Creep.pdf');

% --- Plot Rotation ---
personal_plot_multi(t, rpy_data, ...
    'Time [s]', 'Angle [rad]', ...
    {'Roll', 'Pitch', 'Yaw'}, 'north', 'Rotation_Creep.pdf');

fprintf('Grafici generati ed esportati in PDF.\n');