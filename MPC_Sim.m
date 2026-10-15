%% mpc_single_scenario.m
% MPC tracking of a triangle reference on a fast 2nd-order plant with
% complex-conjugate poles (settling time < 1 s).
% MPC Toolbox is used

clear; clc; close all;

%% -------------------- 1. Plant --------------------
Ts   = 0.01;               % sample time [s]
zeta = 0.7;                % damping ratio -> complex poles, ~5% overshoot
wn   = 8;                  % natural frequency [rad/s]

% Settling time (2%) ~ 4/(zeta*wn) < 1 s -> zeta*wn > 4
Gc    = tf(wn^2, [1, 2*zeta*wn, wn^2]);    % DC gain = 1
plant = c2d(ss(Gc), Ts, 'zoh');

step(plant);

si = stepinfo(plant);
fprintf('Settling time = %.3f s   (overshoot %.1f%%)\n', ...
        si.SettlingTime, si.Overshoot);

% Discrete pole locations
poles_z = eig(plant.A);

fprintf('Discrete pole locations:\n');
for i = 1:length(poles_z)
    fprintf('  z%d = %.6f %+.6fi   |z%d| = %.6f\n', ...
        i, real(poles_z(i)), imag(poles_z(i)), i, abs(poles_z(i)));
end

settling_steps = ceil(si.SettlingTime / Ts);
fprintf('Settling time = %d samples\n\n', settling_steps);

%% -------------------- 2. Reference: triangle, starting at zero --------------------
Tfinal    = 12;
Nsim      = round(Tfinal/Ts);
period    = 4;                    % triangle period [s]
amplitude = 1;                    % peak value
t_ref     = (0:Nsim-1)'*Ts;
r = (amplitude/2) * (sawtooth(2*pi*(1/period)*t_ref, 0.5) + 1);

%% -------------------- 3. MPC controller --------------------
% Prediction horizon must exceed the settling time (~71 samples here).
Np = 50;                   % prediction horizon  (0.80 s lookahead)
Nc = 20;                   % control horizon     (~Np/4)

Wy  = 20;                  % tracking-error weight
Wu  = 0.1;                 % control-effort weight
Wdu = 0.1;                 % control-rate weight

MVmin = -0.6;   MVmax = 0.6;           % input bounds
dMVmin = -0.01; dMVmax = 0.01;      % input rate bounds (per step; = 5 units/s)

mpcobj = mpc(plant, Ts, Np, Nc);
mpcobj.MV.Min     = MVmin;
mpcobj.MV.Max     = MVmax;
mpcobj.MV.RateMin = dMVmin;
mpcobj.MV.RateMax = dMVmax;
mpcobj.Weights.OutputVariables          = Wy;
mpcobj.Weights.ManipulatedVariables     = Wu;
mpcobj.Weights.ManipulatedVariablesRate = Wdu;

%% -------------------- 4. Simulate (receding horizon, with preview) --------------------
simopt = mpcsimopt(mpcobj);
simopt.RefLookAhead = 'on';        % anticipate the upcoming reference

[y, t, u] = sim(mpcobj, Nsim, r, simopt);

%% -------------------- 5. Metrics --------------------
e  = r - y;
du = [0; diff(u)];

RMSE   = sqrt(mean(e.^2));
MaxErr = max(abs(e));
RMSu   = sqrt(mean(u.^2));
MaxU   = max(abs(u));
RMSdu  = sqrt(mean(du.^2));
MaxdU  = max(abs(du));

fprintf('RMSE          = %.4f\n', RMSE);
fprintf('Max error     = %.4f\n', MaxErr);
fprintf('RMS u         = %.4f\n', RMSu);
fprintf('Max |u|       = %.4f\n', MaxU);
fprintf('RMS du        = %.4f\n', RMSdu);
fprintf('Max |du|      = %.4f\n', MaxdU);


%% -------------------- 6. Plots --------------------
desc = sprintf(['Np=%d, Nc=%d  |  Wy=%.3g, Wu=%.3g, Wdu=%.3g  |  ' ...
                'u \\in [%.3g, %.3g], \\Delta u \\in [%.3g, %.3g]  |  preview ON'], ...
                Np, Nc, Wy, Wu, Wdu, MVmin, MVmax, dMVmin, dMVmax);

figure('Name','MPC triangle tracking','Position',[100 100 800 600]);

subplot(2,1,1);
plot(t, r, 'k--', 'LineWidth', 1.3); hold on;
plot(t, y, 'b-', 'LineWidth', 1.6);
xlabel('Time [s]'); ylabel('Output y');
legend('Reference','Output','Location','best'); grid on;
title('MPC Output Tracking');

subplot(2,1,2);
stairs(t, u, 'r-', 'LineWidth', 1.3); hold on;
yline(MVmax, 'k--'); yline(MVmin, 'k--');
xlabel('Time [s]'); ylabel('Input u');
legend('u','MV bounds','Location','best'); grid on;
title('MPC Optimized Control Input');

sgtitle(desc, 'Interpreter', 'tex', 'FontSize', 9);
