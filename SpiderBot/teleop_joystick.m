% =========================================================================
% SCRIPT: teleop_joystick.m
% Controllo SpiderBot con Joystick + Cambio Marcia (Creep/Trot)
% =========================================================================

porta_com = "COM7"; 
baud_rate = 115200;
 
if exist('device', 'var'), clear device; end

try
    device = serialport(porta_com, baud_rate);
    disp("[INFO] Connessione Seriale stabilita!");
    disp("[INFO] Attendo 2.5 secondi che Arduino completi il boot...");
    pause(2.5);
catch
    error("[ERRORE] Impossibile aprire %s.", porta_com);
end

try
    joy = vrjoystick(1); 
catch
    error("[ERRORE] Nessun joystick trovato.");
end

disp("==================================================");
disp(" 🎮 CONTROLLO ATTIVO");
disp(" 🛑 Pulsante 1 (A/X) -> FERMA ED ESCI");
disp(" 🐎 Pulsante 2 (B/O) -> CAMBIA ANDATURA (Creep/Trot)");
disp("==================================================");

freq_invio = 0.05; 
zona_morta = 0.15; 

% Variabili per la gestione dell'andatura
gait_mode = uint8(0); 
btn2_prev = 0;

while true
    [assi, pulsanti, ~] = read(joy);
    
    % --- USCITA (Pulsante 1) ---
    if pulsanti(1) == 1
        invia_pacchetto(device, [0.0; 0.0; 0.0], gait_mode);
        break; 
    end
    
    % --- CAMBIO ANDATURA (Pulsante 2) ---
   % Cambio Andatura con Pulsante 2 (B/O)
    if pulsanti(2) == 1 && btn2_prev == 0
        gait_mode = gait_mode + 1;
        if gait_mode > 1, gait_mode = 0; end % Cicla: 0 -> 1 -> 2 -> 0
        
        switch gait_mode
            case 0, disp("🐢 Mode: CREEP");
            case 1, disp("🐎 Mode: TROT");
        end
    end
    btn2_prev = pulsanti(2);

    
    % --- LETTURA ASSI ---
    v_x     = -assi(2); 
    v_y     = -assi(4); 
    omega_z = -assi(1); 
    
    if abs(v_x) < zona_morta, v_x = 0; end
    if abs(v_y) < zona_morta, v_y = 0; end
    if abs(omega_z) < zona_morta, omega_z = 0; end
    
    V_cmd = [v_x; v_y; omega_z];
    invia_pacchetto(device, V_cmd, gait_mode);
    
    pause(freq_invio); 
end

clear device joy;
disp("[INFO] Programma terminato.");

% =========================================================================
% FUNZIONE MODIFICATA: Ora invia 16 Byte (incluso il gait_mode)
% =========================================================================
function invia_pacchetto(device, V_cmd, gait_mode)
    header = uint8([hex2dec('AA'), hex2dec('55')]);
    
    
    payload_vel = typecast(single(V_cmd), 'uint8');
    payload_vel = payload_vel(:)'; 
    
    % Uniamo le velocità col byte dell'andatura (12 bytes + 1 byte = 13 bytes)
    payload_completo = [payload_vel, gait_mode];
    
    chk = uint8(0);
    for i = 1:length(payload_completo)
        chk = bitxor(chk, payload_completo(i));
    end
    
    pacchetto = [header, payload_completo, chk];
    write(device, pacchetto, "uint8");
end