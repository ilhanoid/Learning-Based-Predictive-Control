%% DeePC_sim.m
% Standalone data-enabled predictive control (DeePC) simulation.
%
% The true state-space model is used ONLY as the simulated physical plant:
%   1) once to collect one offline input-output trajectory, and
%   2) once during the closed-loop experiment.
% The controller itself uses only the Hankel matrices constructed from the
% measured offline input-output data. 
%
% Required toolboxes:
%   - Control System Toolbox (ss, c2d, stepinfo)
%   - Signal Processing Toolbox (sawtooth)
%   - Optimization Toolbox (quadprog)

clear; clc; close all;

%% ==================== 1. Simulated true plant ====================
Ts   = 0.01;
zeta = 0.7;
wn   = 8;

Gc         = tf(wn^2, [1, 2*zeta*wn, wn^2]);
true_plant = c2d(ss(Gc), Ts, 'zoh');
[At, Bt, Ct, Dt] = ssdata(true_plant);
nx = size(At,1);

si = stepinfo(true_plant);
fprintf('TRUE plant: settling %.3f s, overshoot %.1f%%\n', ...
        si.SettlingTime, si.Overshoot);

%% ==================== 2. Reference and constraints ====================
Tfinal    = 12;
Nsim      = round(Tfinal/Ts);
period    = 4;
amplitude = 1;
t         = (0:Nsim-1)'*Ts;
r         = (amplitude/2) * ...
            (sawtooth(2*pi*(1/period)*t, 0.5) + 1);

UMAX  = 0.6;                  % |u(k)|  <= UMAX
DUMAX = 0.01;                 % |du(k)| <= DUMAX per sample

%% ==================== 3. DeePC settings ====================
% Keep the prediction horizon and weights equal to the standalone MPC case
% so that the two simulations can be compared directly.
Np   = 20;                    % future/prediction horizon
Tini = 10;                    % past window; must be at least the system lag

Wy   = 20;                    % output tracking weight
Wu   = 0.1;                   % input magnitude weight
Wdu  = 0.1;                   % input-rate weight


% Hankel representation can give non-unique g vectors. Adding lambda_g*||g||^2
% makes the QP Hessian positive definite and improves numerical conditioning.
lambda_g = 0; % 1e-8;

%% ==================== 4. Collect offline PE data ====================
% Fundamental-lemma requirement for a SISO system:
%   u_data must be persistently exciting of order Tini + Np + nx.
% A sufficient minimum length is
%   Tdata >= (m+1)*(Tini + Np + nx) - 1, with m = 1 here.
% We collect more than this minimum to obtain additional Hankel columns.

% Necessary data-length condition for PE of order Lpe:  H_Lpe(u_data) has m*Lpe rows and Tdata-Lpe+1 columns.
% Full row rank requires:  Tdata-Lpe+1 >= m*Lpe and therefore  Tdata >= (m+1)*Lpe - 1.
%
% For this SISO system, m=1, giving Tdata >= 2*Lpe-1.
% This length only makes full row rank possible; the rank test below verifies whether the generated input is actually persistently exciting.
m       = 1;
Lpe     = Tini + Np + nx;
Tmin    = (m+1)*Lpe - 1;
Tdata   = 65;

if Tdata < Tmin
    error('Tdata=%d is too short. It must be at least %d.', Tdata, Tmin);
end

% The random ramp/dither signal respects both the input and rate bounds.
% It explores large low-frequency motion while its random dither supplies
% broadband richness. 
rng(7);
u_data = generate_rate_limited_pe_input(Tdata, UMAX, DUMAX);

% This loop represents a physical data-collection experiment. At each
% sample one input is applied and one output is measured.
x_data = zeros(nx,1);
y_data = zeros(Tdata,1);
for k = 1:Tdata
    y_data(k) = Ct*x_data + Dt*u_data(k);
    x_data    = At*x_data + Bt*u_data(k);
end

% Verify persistency of excitation at the order required by DeePC.
Hpe       = block_hankel(u_data, Lpe);
tol_pe    = max(size(Hpe))*eps(norm(Hpe,2));
rank_Hpe  = rank(Hpe, tol_pe);

fprintf('Offline data: %d samples (minimum sufficient length: %d)\n', ...
        Tdata, Tmin);
fprintf('PE check: rank(H_%d(u_data)) = %d; required rank = %d\n', ...
        Lpe, rank_Hpe, Lpe);
fprintf('Data limits: max|u| = %.4f, max|du| = %.5f\n\n', ...
        max(abs(u_data)), max(abs(diff(u_data))));

if rank_Hpe < Lpe
    error(['The generated input is not persistently exciting enough. ' ...
           'Increase Tdata or change the excitation seed/design.']);
end

%% ==================== 5. Build the DeePC data matrices ====================
% One Hankel trajectory is divided into a past part (state estimation) and
% a future part (prediction/control):
%
%       [Up]       [u_ini]
%       [Yp] g  =  [y_ini]
%       [Uf]       [u_future]
%       [Yf]       [y_future]

L = Tini + Np;
Hu = block_hankel(u_data, L);
Hy = block_hankel(y_data, L);

Up = Hu(1:Tini, :);
Uf = Hu(Tini+1:end, :);
Yp = Hy(1:Tini, :);
Yf = Hy(Tini+1:end, :);
ng = size(Hu,2);              % number of coefficients in g

% Remove linearly dependent equality rows once. In exact LTI data,
% [Up;Yp] contains redundant rows because the plant order is finite.
Aeq_full = [Up; Yp];
[~, Rqr, row_perm] = qr(Aeq_full', 'vector');
tol_eq = max(size(Aeq_full))*eps(norm(Aeq_full,2));
rank_Aeq = sum(abs(diag(Rqr)) > tol_eq);
independent_rows = sort(row_perm(1:rank_Aeq));
Aeq = Aeq_full(independent_rows,:);

% Difference operator:
% Delta*u_future + d_previous =
% [u(1)-u_previous; u(2)-u(1); ...].
Delta = eye(Np) - diag(ones(Np-1,1), -1);
DUf   = Delta*Uf;

% Terms that do not change during receding-horizon control.
Hqp = 2*(Wy*(Yf'*Yf) + Wu*(Uf'*Uf) + ...
         Wdu*(DUf'*DUf) + lambda_g*eye(ng));
Hqp = (Hqp + Hqp')/2;

Aineq = [ Uf; -Uf; DUf; -DUf ];

qp_options = optimoptions('quadprog', ...
    'Algorithm', 'interior-point-convex', ...
    'Display', 'off');

fprintf('DeePC QP: %d g variables, %d independent past equalities\n\n', ...
        ng, rank_Aeq);

%% ==================== 6. Receding-horizon DeePC ====================
% The zero histories are consistent with the zero initial plant state.
u_ini = zeros(Tini,1);
y_ini = zeros(Tini,1);
u_previous = 0;

x_true = zeros(nx,1);
y      = zeros(Nsim,1);
u      = zeros(Nsim,1);
exitflags = zeros(Nsim,1);

for k = 1:Nsim
    % Reference preview over the entire prediction horizon. Hold the final
    % reference value after the end of the simulation.
    idx = k:min(k+Np-1, Nsim);
    r_horizon = [r(idx); repmat(r(end), Np-numel(idx), 1)];

    % The first predicted rate is u_future(1)-u_previous.
    d_previous = [-u_previous; zeros(Np-1,1)];

    % min 0.5*g'*Hqp*g + fqp'*g
    % The future input and output are Uf*g and Yf*g.
    fqp = -2*Wy*(Yf'*r_horizon) + ...
            2*Wdu*(DUf'*d_previous);

    % Hard input and input-rate constraints.
    bineq = [ UMAX*ones(Np,1); ...
              UMAX*ones(Np,1); ...
              DUMAX*ones(Np,1) - d_previous; ...
              DUMAX*ones(Np,1) + d_previous ];

    beq_full = [u_ini; y_ini];
    beq      = beq_full(independent_rows);

    [g_star, ~, exitflags(k)] = quadprog( ...
        Hqp, fqp, Aineq, bineq, Aeq, beq, [], [], [], qp_options);

    if exitflags(k) <= 0
        error('DeePC QP failed at sample %d (exit flag %d).', ...
              k, exitflags(k));
    end

    u_plan = Uf*g_star;
    u(k)   = u_plan(1);        % receding horizon: apply only first move

    % Simulated physical plant. These matrices are not used by the DeePC QP.
    y(k)   = Ct*x_true + Dt*u(k);
    x_true = At*x_true + Bt*u(k);

    % Shift in the newest measured input-output pair.
    u_ini = [u_ini(2:end); u(k)];
    y_ini = [y_ini(2:end); y(k)];
    u_previous = u(k);
end

%% ==================== 7. Metrics ====================
e  = r-y;
du = [u(1); diff(u)];

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
fprintf('RMS du        = %.5f\n', RMSdu);
fprintf('Max |du|      = %.5f\n', MaxdU);

%% ==================== 8. Plots ====================
t_data = (0:Tdata-1)'*Ts;

figure('Name','Offline data used by DeePC','Position',[80 80 900 620]);
subplot(2,1,1);
stairs(t_data, u_data, 'r-', 'LineWidth',1.1); hold on;
yline( UMAX,'k--'); yline(-UMAX,'k--');
xlabel('Time [s]'); ylabel('u_d'); grid on;
legend('PE data input','input bounds','Location','best');
title(sprintf('Offline excitation: rank H_{%d}(u_d) = %d', ...
              Lpe, rank_Hpe));

subplot(2,1,2);
plot(t_data, y_data, 'b-', 'LineWidth',1.2);
xlabel('Time [s]'); ylabel('y_d'); grid on;
title('Measured offline output');

figure('Name','Standalone DeePC triangle tracking', ...
       'Position',[120 80 900 760]);

subplot(3,1,1);
plot(t, r, 'k--', 'LineWidth',1.3); hold on;
plot(t, y, 'b-', 'LineWidth',1.6);
xlabel('Time [s]'); ylabel('Output y'); grid on;
legend('Reference','DeePC output','Location','best');
title('Output tracking');

subplot(3,1,2);
stairs(t, u, 'r-', 'LineWidth',1.3); hold on;
yline( UMAX,'k--'); yline(-UMAX,'k--');
xlabel('Time [s]'); ylabel('Input u'); grid on;
legend('DeePC input','input bounds','Location','best');
title('Control input');

subplot(3,1,3);
stairs(t, du, 'Color',[0.2 0.55 0.2], 'LineWidth',1.2); hold on;
yline( DUMAX,'k--'); yline(-DUMAX,'k--');
xlabel('Time [s]'); ylabel('\Delta u'); grid on;
legend('\Delta u','rate bounds','Location','best');
title('Control-input rate');

sgtitle(sprintf(['Standalone DeePC: N=%d, T_{ini}=%d, ' ...
                 'W_y=%.3g, W_u=%.3g, W_{du}=%.3g'], ...
                 Np, Tini, Wy, Wu, Wdu));

%% ==================== Local functions ====================
function H = block_hankel(signal, depth)
%BLOCK_HANKEL Scalar block-Hankel matrix with the requested depth.
    signal = signal(:);
    T = numel(signal);
    ncols = T-depth+1;
    if ncols < 1
        error('Signal length must be at least the Hankel depth.');
    end

    H = zeros(depth,ncols);
    for row = 1:depth
        H(row,:) = signal(row:row+ncols-1).';
    end
end

function u = generate_rate_limited_pe_input(T, UMAX, DUMAX)
%GENERATE_RATE_LIMITED_PE_INPUT Bounded random ramps plus small dither.
% The long random ramps give useful output amplitude. Random switching times
% and sample-wise dither add richness while respecting |u| and |du|.
    u = zeros(T,1);
    direction = 1;
    next_switch = 1;
    soft_limit = 0.85*UMAX;

    for k = 2:T
        if k >= next_switch
            direction = 2*(rand > 0.5)-1;
            next_switch = k + randi([15,70]);
        end

        step = direction*0.75*DUMAX + ...
               0.20*DUMAX*(2*rand-1);
        candidate = u(k-1) + step;

        if candidate > soft_limit
            candidate = soft_limit;
            direction = -1;
            next_switch = k + randi([15,70]);
        elseif candidate < -soft_limit
            candidate = -soft_limit;
            direction = 1;
            next_switch = k + randi([15,70]);
        end

        u(k) = candidate;
    end

    % Remove DC bias, then scale conservatively if centering brought the
    % signal close to an amplitude or rate limit.
    u = u-mean(u);
    amplitude_scale = 0.90*UMAX/max(max(abs(u)),eps);
    rate_scale = 0.90*DUMAX/max(max(abs(diff(u))),eps);
    u = min([1, amplitude_scale, rate_scale])*u;
end
