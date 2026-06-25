%% Script Principale
close all; clc;

% Calcolo tempo
num_samples = size(out.pos, 3); 
t_data = linspace(0, sim_time, num_samples)';

% Estrazione dati (Tua logica originale)
pos_data = squeeze(out.pos)'; 
vel_data = squeeze(out.vel)';
rot_data = squeeze(out.rotation)'; 

% --- 1. Plot Posizione ---
personal_plot_style(t_data, pos_data(:, 1:3), ...
    'Time [s]', 'Position [m]', ...
    {'x', 'y', 'z'}, 'north', 'Position_Pace.pdf');

% --- 2. Plot Velocità ---
personal_plot_style(t_data, vel_data(:, 1:3), ...
    'Time [s]', 'Velocity [m/s]', ...
    {'v_x', 'v_y', 'v_z'}, 'north', 'Velocity_Pace.pdf');

% --- 3. Plot Rotazione ---
personal_plot_style(t_data, rot_data(:, 1:3), ...
    'Time [s]', 'Angle [rad]', ...
    {'Roll', 'Pitch', 'Yaw'}, 'north', 'Rotation_Pace.pdf');