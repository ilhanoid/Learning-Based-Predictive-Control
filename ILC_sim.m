% ILC (Iterative Learning Control) tracking of a triangle reference on the
% same fast 2nd-order plant used in the MPC script.
%
% Learning law: PD-type with zero-phase Q-filter
%
%   u_{j+1}(k) = Q(q)[ u_j(k) + kp*e_j(k+1) + kd*(e_j(k+1) - e_j(k)) ]
%
% Unlike MPC, ILC has no constraint handling: it converges to whatever
% input achieves tracking, which may violate actuator limits

clear; clc; close all;

%% -------------------- 1. Plant (same as the MPC study) --------------------
Ts   = 0.01;               % sample time [s]
zeta = 0.7;                % damping ratio -> complex poles, ~5% overshoot
wn   = 8;                  % natural frequency [rad/s]

Gc    = tf(wn^2, [1, 2*zeta*wn, wn^2]);    % DC gain = 1
plant = c2d(ss(Gc), Ts, 'zoh');
[A,B,C,D] = ssdata(plant);
nx = size(A,1);

si = stepinfo(plant);
fprintf('Settling time = %.3f s   (overshoot %.1f%%)\n', si.SettlingTime, si.Overshoot);
fprintf('Discrete poles: '); disp(eig(A).');

%% -------------------- 2. Reference: triangle + zero padding --------------------
% Same plant and same triangle as the MPC script
% Reference is zero padded at both  ends. 
% Padding is needed because:
%   a) the ILC update is noncausal -- it uses e(k+1), and the zero-phase
%       Q-filter (filtfilt) runs the data forwards & backward --> both edges need real margin 
%   b) the plant must be allowed to settle inside the trial window.


Tactive   = 12;                    % duration of the actual trajectory [s]
period    = 4;                     % triangle period [s]
amplitude = 1;                     % peak value

Nactive = round(Tactive/Ts);            % active time step number
Npad    = ceil(si.SettlingTime/Ts);   % pad >= settling time ( add to each side)
N       = Nactive + 2*Npad;             % total samples per ILC trial
t       = (0:N-1)'*Ts;

t_active = (0:Nactive-1)'*Ts;
r_active = (amplitude/2) * (sawtooth(2*pi*(1/period)*t_active, 0.5) + 1);
r        = [zeros(Npad,1); r_active; zeros(Npad,1)]; % 0000--ref--0000

act = (Npad+1) : (Npad+Nactive); % index range of the actual trajectory inside the padded trial


fprintf('\nTrial length: %d samples (%d active + %d pad each side)\n', N, Nactive, Npad);

%% -------------------- 3. ILC learning law --------------------
% The convergence factor of a causal PD-type ILC is |1 - (kp+kd)*p1|,
% where p1 = C*B is the plant's first Markov parameter. Here p1 is very
% small (fine sample time + relative-degree-2 plant), so the learning
% gains must be large. ---> but with large gains the rmse diverged
%p1 = C*B;
%gamma_target = 0.5;        % desired error reduction per iteration (0 < g < 1) 

%l0 = (1 - gamma_target) / p1;
kp = 0.266;
kd =  4;

%gamma = abs(1 - (kp + kd) * p1);
%fprintf('\nFirst Markov parameter p1 = C*B = %.6g\n', p1);
%fprintf('Stability requires 0 < kp+kd < %.1f\n', 2/p1);
%fprintf('Chosen kp = %.1f, kd = %.1f  ->  convergence factor = %.3f', kp, kd, gamma);
% if gamma < 1
%     fprintf('  (stable)\n\n');
% else
%     fprintf('  (UNSTABLE -- lower the gains)\n\n');
% end
% -------------------------
% Zero-phase low-pass Q-filter: improves robustness and suppresses
% high-frequency learning transients. Applied offline via filtfilt, so it adds no phase lag 

useQ      = true;
Qcutoff   = 2;            % [Hz]  (plant bandwidth ~ wn/2pi = 1.3 Hz) 
if useQ
    Qfilt = designfilt('lowpassfir', 'FilterOrder', 20, ...          
                       'CutoffFrequency', Qcutoff, 'SampleRate', 1/Ts);
end

Ntrials = 20;              % number of learning iterations

%% -------------------- 4. ILC iteration loop --------------------
% run trial → measure error → update input → next trial
u       = zeros(N, Ntrials+1);   %% stores input for each time step in each trial % u(:,1) = trial 0  
rmse    = zeros(Ntrials,1);      % store the tracking error metrics for each trial
maxerr  = zeros(Ntrials,1);
Y       = zeros(N, Ntrials);

for j = 1:Ntrials
    % --- run one trial --> initial conditions reset to zeros(nx,1) each trial 
    y = lsim(plant, u(:,j), t, zeros(nx,1));
    Y(:,j) = y;

    e = r - y;
    rmse(j)   = sqrt(mean(e(act).^2));    % scored on the active region only
    maxerr(j) = max(abs(e(act)));

    % --- PD-type update (uses e(k+1), i.e. one-step-ahead error) ---
    

    e_next = [e(2:end); 0];               % e_j(k+1)
    de     = [diff(e); 0];                % e_j(k+1) - e_j(k)
    u_new  = u(:,j) + kp*e_next + kd*de;

    if useQ
        u_new = filtfilt(Qfilt, u_new);   % zero-phase filtering
    end

    u(:,j+1) = u_new;
end

%% -------------------- 5. Metrics --------------------
y_first = Y(:,1);
y_last  = Y(:,end);
u_final = u(:,Ntrials);

e_last  = r - y_last;
du      = [0; diff(u_final)];

% Metrics reported over the active trajectory only (padding excluded), so the leading/trailing zeros don't flatter the averages.
fprintf('\n(metrics over the active trajectory, padding excluded)\n');
fprintf('Trial 1   : RMSE = %.4f,  Max error = %.4f\n', rmse(1),   maxerr(1));
fprintf('Trial %-3d : RMSE = %.4f,  Max error = %.4f\n', Ntrials, rmse(end), maxerr(end));
fprintf('Improvement: %.1f%% reduction in RMSE\n\n', 100*(1 - rmse(end)/rmse(1)));

fprintf('Converged input:  max|u| = %.4f,  RMS u = %.4f\n', max(abs(u_final(act))), sqrt(mean(u_final(act).^2)));
fprintf('                  max|du| = %.4f per step (= %.2f units/s)\n', max(abs(du(act))), max(abs(du(act)))/Ts);

%%-------- check constraints
MVmax_ref = 7; dMVmax_ref = 0.05;
fprintf('\nAgainst the MPC study''s limits (|u|<=%.2f, |du|<=%.2f/step):\n', MVmax_ref, dMVmax_ref);
if max(abs(u_final(act))) > MVmax_ref
    fprintf('  -> input bound VIOLATED by ILC (%.2f > %.2f)\n', max(abs(u_final(act))), MVmax_ref);
else
    fprintf('  -> input bound respected (%.2f <= %.2f)\n', max(abs(u_final(act))), MVmax_ref);
end
if max(abs(du(act))) > dMVmax_ref
    fprintf('  -> rate bound VIOLATED by ILC (%.4f > %.4f)\n', max(abs(du(act))), dMVmax_ref);
else
    fprintf('  -> rate bound respected (%.4f <= %.4f)\n', max(abs(du(act))), dMVmax_ref);
end

%% -------------------- 6. Plots --------------------
desc = sprintf(['ILC PD-type  |  kp=%.1f, kd=%.1f  |   ' ...
                'Q-filter %d Hz  |  %d trials  |  zero pad %d samples'], ...
                kp, kd, Qcutoff, Ntrials, Npad);

figure('Name','ILC convergence and tracking','Position',[100 100 850 700]);

subplot(3,1,1);
semilogy(1:Ntrials, rmse, 'o-', 'LineWidth', 1.4); hold on;
semilogy(1:Ntrials, maxerr, 's-', 'LineWidth', 1.4);
xlabel('Iteration j'); ylabel('Error');
legend('RMS error','Max error','Location','best'); grid on;
title('Learning Curve');

subplot(3,1,2);
plot(t, r, 'k--', 'LineWidth', 1.3); hold on;
plot(t, y_first, 'r:',  'LineWidth', 1.2);
plot(t, y_last,  'b-',  'LineWidth', 1.6);
xline(t(act(1)),   'k:', 'HandleVisibility','off');   % zero-pad boundaries
xline(t(act(end)), 'k:', 'HandleVisibility','off');
xlabel('Time [s]'); ylabel('Output y');
legend('Reference', 'Trial 1', sprintf('Trial %d', Ntrials), 'Location','best');
grid on; title('ILC Output Tracking Before vs. After Learning ');

subplot(3,1,3);
stairs(t, u_final, 'r-', 'LineWidth', 1.3); hold on;
xline(t(act(1)),   'k:', 'HandleVisibility','off');
xline(t(act(end)), 'k:', 'HandleVisibility','off');
xlabel('Time [s]'); ylabel('Input u');
legend('Learned input (final trial)','Location','best'); grid on;
title('Converged Control Input ');

sgtitle(desc, 'Interpreter', 'tex', 'FontSize', 9);