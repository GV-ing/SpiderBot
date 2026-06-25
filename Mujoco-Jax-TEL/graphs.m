%% Script Principale: Analisi Telemetria Goal-Conditioned (Stile LaTeX)
close all; clear; clc;

% 1. Caricamento dati esportati da Python (enjoy_spider.py)
% Usiamo readtable per leggere direttamente il file CSV generato
if isfile('telemetry.csv')
    dati = readtable('telemetry.csv');
else
    error('File telemetry.csv non trovato. Esegui prima enjoy_spider.py e chiudi la finestra 3D per generarlo.');
end

% Estrazione asse dei tempi
t = dati.time;

% 2. Organizzazione dei dati in matrici [N_samples x 2] (Comando vs Reale)
% Ogni matrice affianca il comando del joystick alla risposta reale del robot

% Asse X (Avanzamento longitudinale)
data_x = [dati.cmd_vx, dati.real_vx];

% Asse Y (Traslazione laterale / Strafe)
data_y = [dati.cmd_vy, dati.real_vy];

% Asse Yaw (Rotazione sul posto)
data_wz = [dati.cmd_wz, dati.real_wz];

%% 3. Generazione Grafici tramite la funzione personalizzata

% --- Plot Asse X (Velocità Lineare X) ---
personal_plot_multi(t, data_x, ...
    'Time [s]', 'Velocity X [m/s]', ...
    {'Cmd v_x', 'Real v_x'}, 'northeast', 'Tracking_Vx.pdf');

% --- Plot Asse Y (Velocità Lineare Y) ---
personal_plot_multi(t, data_y, ...
    'Time [s]', 'Velocity Y [m/s]', ...
    {'Cmd v_y', 'Real v_y'}, 'northeast', 'Tracking_Vy.pdf');

% --- Plot Asse Yaw (Velocità Angolare Z) ---
personal_plot_multi(t, data_wz, ...
    'Time [s]', 'Angular Vel Yaw [rad/s]', ...
    {'Cmd \omega_z', 'Real \omega_z'}, 'northeast', 'Tracking_Yaw.pdf');

fprintf('Grafici di telemetria generati ed esportati in PDF con successo.\n');