% Main idea: ILC learns accurately but ignores constraints. DeePC uses the
% measured ILC input-output trajectories directly, without identifying a
% model, to produce a constraint-feasible reference for ILC.

% Combined ILC + DeePC scheme:
%     -> Learn one fixed FIR inverse from filtered-impulse tracking
%     -> Reuse that inverse for triangular-reference ILC on the true plant
%     -> Collect u_ILC and y_ILC
%     -> Build full Hankel matrices directly from all ILC input-output data
%     -> Compress their column space using one economy SVD
%     -> Solve DeePC in the compressed trajectory-coordinate vector
%     -> DeePC output becomes the feasible (attainable) reference
%     -> Use that feasible reference for ILC
%     -> Replace the old dataset with the latest ILC period, rebuild, repeat


%% Implementation Details
%
% ITERATIVE LEARNING CONTROL (ILC) -
%   • A filtered impulse is first tracked on the true plant using the original
%     fixed PD-type learning law as L0.
%   • After this identification phase converges, the learned impulse input
%     is divided by its scale alpha and stored as the FIR inverse f_final.
%   • f_final is then reused for every triangular / DeePC-shaped reference:
%         u0      = f_final * r
%         u_{j+1} = u_j + f_final * e_j
%    * denotes centered, noncausal FIR convolution.
%   • f_final is fixed after impulse identification, it is not updated inside later trials.
%   • Initial plant states are reset before every trial.
%   • Performance metrics are evaluated over the unpadded region only.
%
% DeePC
%   • At each outer iteration, the previous Hankel dataset is discarded.
%   • The new matrices span the complete latest ILC input-output trajectory,
%     including all triangle periods and the zero-padding regions.
%   • Every one of the 1321 overlapping windows is included before compression.
%   • SVD replaces the 1321 raw coefficients by the retained
%     orthogonal trajectory directions. No time samples or periods are removed.
%   • The prediction is u_future = Uf*z and y_future = Yf*z, where z is the
%     compressed DeePC decision vector.
%   • The QP retains the past equality, hard
%     input/rate constraints, and first-move receding-horizon operation.
%
% EXPERIMENTAL TOGGLES
%   FIRST_PASS_SCALE = 1          Disable initial reference scaling.
%   ADD_DEEPC_DATA_NOISE = false  Disable measurement noise in Hankel data.
%   ADD_INVERSE_ID_NOISE = true  Disable noise during inverse identification.                      
%



clear; clc; close all;

%% ==================== 1. TRUE plant ====================
Ts   = 0.01;
zeta = 0.7;
wn   = 8;
Gc         = tf(wn^2, [1, 2*zeta*wn, wn^2]);
true_plant = c2d(ss(Gc), Ts, 'zoh');
[At,Bt,Ct,~] = ssdata(true_plant);
nx = size(At,1);

si = stepinfo(true_plant);
fprintf('TRUE plant: settling %.3f s, overshoot %.1f%%\n', si.SettlingTime, si.Overshoot);

%% ==================== 2. Reference (triangle + zero padding) ====================
Tactive   = 12;
period    = 4;
amplitude = 1;

Nactive = round(Tactive/Ts);
Npad    = ceil(si.SettlingTime/Ts); %Tactive/Ts;
N       = Nactive + 2*Npad;
t       = (0:N-1)'*Ts;

t_active = (0:Nactive-1)'*Ts;
r_active = (amplitude/2) * (sawtooth(2*pi*(1/period)*t_active, 0.5) + 1);
r_orig   = [zeros(Npad,1); r_active; zeros(Npad,1)];   % original desired trajectory
act      = (Npad+1) : (Npad+Nactive);                  % active (unpadded) region

%% ==================== 3. Constraints & Noise ====================
% Chosen such that the input is bounded
UMAX  = 0.6;        % |u|  <= UMAX
DUMAX = 0.010;      % |du| <= DUMAX per step
ADD_DEEPC_DATA_NOISE = false; % measurement noise in physical ILC output data
noise_std = 0.010;
rng(1);                         % reproducible measurement-noise sequence

fprintf('Constraints: |u| <= %.3f, |du| <= %.4f/step  \n', ...
        UMAX, DUMAX);

%% ==================== 4. Inverse identification and ILC settings ====================
% First learn a reusable, band-limited FIR inverse from a
% filtered-impulse experiment and keeps
% the initial learning filter L0 fixed throughout the identification phase.

Mcutoff = 2;             % [Hz] bandwidth over which the inverse is learned
Morder  = 20;            % order of the FIR low-pass reference model M(z)
INV_ALPHA = 1.0;         % impulse scale: increase for SNR, decrease for input safety
NinverseTrials = 10;     % fixed-PD impulse-learning trials
Ntrials = 10;            % inverse-filter ILC trials per outer iteration

% Original PD-type ILC parameters. These are used only to learn the filtered
% impulse input. After that phase, f_final replaces the PD learning law.
kp      = 0.266;
kd      = 4;
Qcutoff = 5;             % [Hz] zero-phase filter
ADD_INVERSE_ID_NOISE = true;
inverse_noise_std = 0.010;

Mfilt = designfilt('lowpassfir', 'FilterOrder', Morder, ...
                   'CutoffFrequency', Mcutoff, 'SampleRate', 1/Ts);

Qfilt = designfilt('lowpassfir', 'FilterOrder', 20, ...
                   'CutoffFrequency', Qcutoff, 'SampleRate', 1/Ts);

% Optional safety scaling of the FIRST ILC pass.
% The first ILC run can use r_safe = FIRST_PASS_SCALE * r_orig so that
% the initial data collection for DeePC stays within input
% limits. Set FIRST_PASS_SCALE = 1.0 to disable scaling and use the full
% original reference from the beginning.
FIRST_PASS_SCALE = 0.5;

%% ==================== 5. Learn a fixed FIR inverse from a filtered impulse ====================
% A centered impulse permits preview/noncausal inverse coefficients on both sides of zero lag.
% Filtering it avoids attempting unsafe high-frequency inversion. The learned input divided by INV_ALPHA is the FIR inverse.
inv_center = floor((N+1)/2);
delta = zeros(N,1);
delta(inv_center) = 1;
r_inverse = INV_ALPHA * filtfilt(Mfilt, delta);

[f_final, u_inverse, y_inverse, inverse_relerr] = ...
    identify_inverse_fir_pd(true_plant, r_inverse, INV_ALPHA, ...
        kp, kd, Qfilt, t, NinverseTrials, nx, ...
        ADD_INVERSE_ID_NOISE, inverse_noise_std);

fprintf(['Inverse ID: %d fixed-PD trials | final relative impulse error = %.4e ' ...
         '| max|u_imp| = %.4f\n\n'], ...
        NinverseTrials, inverse_relerr(end), max(abs(u_inverse)));

%% ==================== 6. Full-data SVD-DeePC settings ====================
% These values and the QP below match deepc_single_scenario.m.
Np   = 20;
Tini = 10;
Wy   = 20;
Wu   = 0.1;
Wdu  = 0.1;
lambda_g = 0;

% Build the raw Hankel matrices from all 1350 samples. With depth
% Tini+Np=30, all 1321 overlapping windows are used. Their joint behavior
% matrix has only 2*(Tini+Np)=60 rows, so an economy SVD reduces the QP from
% 1321 raw coefficients to at most 60 trajectory coordinates.
%
% Directions below SVD_REL_TOL*sigma_max are discarded. The numerical-rank
% tolerance is also enforced. A small relative tolerance preserves the exact
% noiseless behavior; increase it cautiously when output data are noisy.
SVD_REL_TOL = 1e-10;

%% ==================== 7. OUTER LOOP ====================
NouterMax = 2;

% Initial ILC target.
% If FIRST_PASS_SCALE < 1, ILC first tracks a scaled-down reference.

r_ilc = FIRST_PASS_SCALE * r_orig;

fprintf('Initial ILC reference scale = %.2f (1.00 = no scaling)\n', FIRST_PASS_SCALE);

hist = struct('rmse_vs_ref',[], 'rmse_vs_orig',[], 'maxu',[], 'maxdu',[], ...
             'deepc_rmse',[], 'deepc_maxu',[], 'deepc_maxdu',[], ...
             'input_rank',[], 'behavior_rank',[], 'ng',[], ...
             'viol_u',[], 'viol_du',[]);
R_hist = zeros(N, NouterMax);     % stores the reference trajectory used by ILC at each outer iteration (feasible reference)
Y_hist = zeros(N, NouterMax);     % stores the ILC output at each outer iteration
U_hist = zeros(N, NouterMax);     % stores the final learned ILC input at each outer iteration
Udeepc_hist = zeros(N, NouterMax);% stores the DeePC-optimized input
Yfeas_hist  = zeros(N, NouterMax);% stores DeePC output (= next ILC target)

for outer = 1:NouterMax
    fprintf('=========== OUTER ITERATION %d ===========\n', outer);
    
    % ---------- 7a. Inverse-filter ILC on the true plant ----------
    % For each new reference, f_final supplies both the initial feedforward input and subsequent
    % trial-to-trial error correction. The true plant still generates every ILC output.
    [u_ilc, y_ilc, rmse_curve] = run_inverse_ilc( ...
        true_plant, r_ilc, t, act, f_final, inv_center, Ntrials, nx);

    du_ilc   = [0; diff(u_ilc)];
    maxu     = max(abs(u_ilc(act)));
    maxdu    = max(abs(du_ilc(act)));
    e_ref    = r_ilc  - y_ilc;   % error relative to current ILC target
    e_orig   = r_orig - y_ilc;   % error relative to original desired reference

    fprintf(['  ILC : RMSE vs current ref = %.4f | ' ...
             'RMSE vs ORIGINAL = %.4f\n'], ...
            sqrt(mean(e_ref(act).^2)), sqrt(mean(e_orig(act).^2)));
    fprintf('        max|u| = %.4f %s | max|du| = %.5f %s\n', ...
            maxu,  tern(maxu  > UMAX,  '(VIOLATES)', '(ok)'), ...
            maxdu, tern(maxdu > DUMAX, '(VIOLATES)', '(ok)'));

    % ---------- 7b. Replace Hankel data with latest complete ILC trial ----------
    % Use the complete latest physical ILC trajectory. Earlier outer-iteration
    % datasets are not accumulated. Measurement noise, when enabled, is added
    % to the complete measured output before the Hankel windows are formed.
    u_deepc_data = u_ilc;
    y_deepc_data = y_ilc;
    if ADD_DEEPC_DATA_NOISE
        y_deepc_data = y_deepc_data + noise_std*randn(size(y_deepc_data));
    end

    [Up, Yp, Uf, Yf, input_rank, behavior_rank, ng, ...
        raw_columns, singular_values] = ...
        build_svd_ilc_hankel(u_deepc_data, y_deepc_data, ...
            Tini, Np, nx, SVD_REL_TOL);

    fprintf(['  DATA: latest outer iteration only, complete trajectory, ' ...
             '%d samples\n'], numel(u_deepc_data));
    fprintf(['        all %d raw Hankel columns used before SVD; ' ...
             'compressed decision dimension = %d\n'], raw_columns, ng);
    fprintf(['        sigma_1 = %.3e; sigma_%d/sigma_1 = %.3e; ' ...
             'SVD relative tolerance = %.1e\n'], ...
            singular_values(1), ng, ...
            singular_values(ng)/singular_values(1), SVD_REL_TOL);
    fprintf(['        rank H_%d(u) = %d/%d; behavior rank = %d ' ...
             '(expected %d)\n'], Tini+Np+nx, input_rank, ...
            Tini+Np+nx, behavior_rank, Tini+Np+nx);
    if input_rank < Tini+Np+nx
        fprintf('        PE is not strictly satisfied; DeePC continues by design.\n');
    end

    % ---------- 7c. SVD-coordinate DeePC: generate feasible reference -------
    % The standalone objective and constraints are unchanged. Only the
    % trajectory coordinates have been compressed; the controller still uses
    % exact past matching and applies only the first optimized input.
    [y_feasible, u_deepc, exitflags] = run_svd_deepc( ...
        true_plant, r_orig, Up, Yp, Uf, Yf, ...
        Tini, Np, Wy, Wu, Wdu, lambda_g, UMAX, DUMAX);

    du_deepc = [u_deepc(1); diff(u_deepc)];
    deepc_rmse = sqrt(mean((r_orig(act)-y_feasible(act)).^2));
    fprintf(['  DeePC: feasible-ref RMSE vs ORIGINAL = %.4f | ' ...
             'max|u| = %.4f | max|du| = %.5f | QP failures = %d\n'], ...
            deepc_rmse, max(abs(u_deepc(act))), ...
            max(abs(du_deepc(act))), sum(exitflags <= 0));

    % ---------- 7d. logging ----------
    hist.rmse_vs_ref(outer)  = sqrt(mean(e_ref(act).^2));
    hist.rmse_vs_orig(outer) = sqrt(mean(e_orig(act).^2));
    hist.maxu(outer)         = maxu;
    hist.maxdu(outer)        = maxdu;
    hist.deepc_rmse(outer)   = deepc_rmse;
    hist.deepc_maxu(outer)   = max(abs(u_deepc(act)));
    hist.deepc_maxdu(outer)  = max(abs(du_deepc(act)));
    hist.input_rank(outer)   = input_rank;
    hist.behavior_rank(outer)= behavior_rank;
    hist.ng(outer)           = ng;
    hist.viol_u(outer)       = maxu  > UMAX;
    hist.viol_du(outer)      = maxdu > DUMAX;

    R_hist(:,outer)     = r_ilc;
    Y_hist(:,outer)     = y_ilc;
    U_hist(:,outer)     = u_ilc;
    Udeepc_hist(:,outer)= u_deepc;
    Yfeas_hist(:,outer) = y_feasible;

    % ---------- 7e. feasible reference becomes the next ILC target ----------
    r_ilc = y_feasible;
    fprintf('\n');
end

%% ==================== 8. Summary ====================
T = table((1:NouterMax)', hist.rmse_vs_ref(:), hist.rmse_vs_orig(:), ...
          hist.maxu(:), hist.maxdu(:), hist.deepc_rmse(:), ...
          hist.deepc_maxu(:), hist.deepc_maxdu(:), hist.ng(:), ...
          hist.input_rank(:), hist.behavior_rank(:), ...
          'VariableNames', {'Outer','RMSE_vs_ref','RMSE_vs_orig','MaxU','MaxdU', ...
                            'DeePC_RMSE','DeePC_MaxU','DeePC_MaxdU', ...
                            'SVDDecisionDimension','InputHankelRank','BehaviorRank'});
disp(T);

fprintf('Constraint status of the ILC input, per outer iteration:\n');
for i = 1:NouterMax
    fprintf('  iter %d: |u|=%.4f %s   |du|=%.5f %s\n', i, ...
        hist.maxu(i),  tern(hist.viol_u(i),  'VIOLATES', 'ok'), ...
        hist.maxdu(i), tern(hist.viol_du(i), 'VIOLATES', 'ok'));
end

%% ==================== 9. Plots ====================

figure('Name','Reference reshaping','Position',[120 120 950 650]);

subplot(2,1,1);
plot(t, r_orig, 'k--', 'LineWidth',1.6); hold on;
plot(t, R_hist(:,min(2,NouterMax)), 'b-',  'LineWidth',1.3);
plot(t, R_hist(:,NouterMax), 'r-', 'LineWidth',1.3);
xlabel('Time [s]'); ylabel('Reference'); grid on;
legend('original (desired)','feasible, iter 2','feasible, final','Location','best');
title('DeePC reshapes the reference into an attainable trajectory');

subplot(2,1,2);
stairs(t, U_hist(:,1), 'b-', 'LineWidth',1.2); hold on;
stairs(t, U_hist(:,NouterMax), 'r-', 'LineWidth',1.3);
yline( UMAX,'k--','LineWidth',1.2); yline(-UMAX,'k--','LineWidth',1.2);
xlabel('Time [s]'); ylabel('Input u'); grid on;
legend('ILC input, iter 1','ILC input, final','constraints','Location','best');
title('ILC input before vs. after reference reshaping');

%% ---- DeePC vs ILC: final completed DeePC-to-ILC pair ----

kILC   = NouterMax;
kDeePC = max(1,NouterMax-1);

figure('Name','DeePC vs ILC: inputs and outputs','Position',[150 60 1150 720]);

subplot(2,2,1);
plot(t, r_orig, 'k--', 'LineWidth',1.4); hold on;
plot(t, Yfeas_hist(:,kDeePC), 'b-', 'LineWidth',1.5);
xlabel('Time [s]'); ylabel('Output'); grid on;
legend('original reference','DeePC output = feasible ref','Location','best');
title(sprintf('DeePC stage - output (iter %d)', kDeePC));

subplot(2,2,2);
stairs(t, Udeepc_hist(:,kDeePC), 'b-', 'LineWidth',1.3); hold on;
yline( UMAX,'k--','LineWidth',1.2); yline(-UMAX,'k--','LineWidth',1.2);
xlabel('Time [s]'); ylabel('u_{DeePC}'); grid on;
legend('DeePC optimized input','constraint','Location','best');
title(sprintf('DeePC stage - input   (max|u| = %.4f)', ...
      max(abs(Udeepc_hist(act,kDeePC)))));

subplot(2,2,3);
plot(t, R_hist(:,kILC), 'k--', 'LineWidth',1.4); hold on;
plot(t, Y_hist(:,kILC), 'r-', 'LineWidth',1.5);
plot(t, r_orig, 'Color',[.6 .6 .6], 'LineStyle',':', 'LineWidth',1.1);
xlabel('Time [s]'); ylabel('Output'); grid on;
legend('feasible ref (ILC target)','ILC output','original ref','Location','best');
title(sprintf('ILC stage - output (iter %d)', kILC));

subplot(2,2,4);
stairs(t, U_hist(:,kILC), 'r-', 'LineWidth',1.3); hold on;
yline( UMAX,'k--','LineWidth',1.2); yline(-UMAX,'k--','LineWidth',1.2);
xlabel('Time [s]'); ylabel('u_{ILC}'); grid on;
legend('ILC learned input','constraint','Location','best');
title(sprintf('ILC stage - input   (max|u| = %.4f %s)', ...
      max(abs(U_hist(act,kILC))), ...
      tern(max(abs(U_hist(act,kILC)))>UMAX,'VIOLATES','ok')));

sgtitle('Full-data SVD-DeePC generates the feasible reference; ILC then learns to track it');

%% ==================== local functions ====================
%% Build full-data Hankel matrices and compress their coefficient space by SVD
function [Up, Yp, Uf, Yf, input_rank, behavior_rank, ng, ...
          raw_columns, singular_values] = ...
    build_svd_ilc_hankel(u_data, y_data, Tini, Np, nx, svd_relative_tol)

    L   = Tini + Np;
    Lpe = L + nx;

    u_data = u_data(:);
    y_data = y_data(:);
    if numel(u_data) ~= numel(y_data)
        error('The latest ILC input and output data have different lengths.');
    end
    if numel(u_data) < Lpe
        error('The latest ILC trajectory is shorter than Tini+Np+nx.');
    end

    % Construct every overlapping input-output window from all physical data.
    Hu = block_hankel(u_data, L);
    Hy = block_hankel(y_data, L);
    Hpe = block_hankel(u_data, Lpe);
    raw_columns = size(Hu,2);

    Up_raw = Hu(1:Tini,:);
    Uf_raw = Hu(Tini+1:end,:);
    Yp_raw = Hy(1:Tini,:);
    Yf_raw = Hy(Tini+1:end,:);

    % Stack the complete raw behavior before performing the SVD. V contains
    % orthonormal combinations of all 1321 original Hankel columns:
    %
    %   Hbehavior = U*S*V'
    %   g_raw     = V_r*z
    %   Hbehavior*g_raw = (Hbehavior*V_r)*z.
    %
    % Therefore z replaces g_raw in the QP without selecting one period or
    % discarding particular time regions.
    Hbehavior = [Up_raw; Yp_raw; Uf_raw; Yf_raw];
    [~, Sbehavior, Vbehavior] = svd(Hbehavior,'econ');
    singular_values = diag(Sbehavior);

    if isempty(singular_values) || singular_values(1) <= 0
        error('The complete ILC Hankel behavior matrix contains no usable data.');
    end

    numerical_tol = max(size(Hbehavior))*eps(singular_values(1));
    retained_tol = max(numerical_tol, svd_relative_tol*singular_values(1));
    ng = sum(singular_values > retained_tol);
    if ng == 0
        error('The SVD tolerance removed every DeePC trajectory direction.');
    end

    Vretained = Vbehavior(:,1:ng);
    Up = Up_raw*Vretained;
    Yp = Yp_raw*Vretained;
    Uf = Uf_raw*Vretained;
    Yf = Yf_raw*Vretained;

    % Report the numerical ranks of the full, uncompressed data matrices.
    singular_input = svd(Hpe,'econ');
    input_tol = max(size(Hpe))*eps(singular_input(1));
    input_rank = sum(singular_input > input_tol);
    behavior_rank = sum(singular_values > numerical_tol);
end

%% DeePC controller in the retained SVD trajectory coordinates
function [y, u, exitflags] = run_svd_deepc( ...
    plant, reference, Up, Yp, Uf, Yf, ...
    Tini, Np, Wy, Wu, Wdu, lambda_g, UMAX, DUMAX)

    [At, Bt, Ct, Dt] = ssdata(plant);
    nx = size(At,1);
    Nsim = numel(reference);
    ng = size(Uf,2);

    % Retain independent equality rows exactly as in standalone DeePC.
    Aeq_full = [Up; Yp];
    [~, Rqr, row_perm] = qr(Aeq_full', 'vector');
    tol_eq = max(size(Aeq_full))*eps(norm(Aeq_full,'fro'));
    rank_Aeq = sum(abs(diag(Rqr)) > tol_eq);
    independent_rows = sort(row_perm(1:rank_Aeq));
    Aeq = Aeq_full(independent_rows,:);

    Delta = eye(Np) - diag(ones(Np-1,1),-1);
    DUf = Delta*Uf;

    Hqp = 2*(Wy*(Yf'*Yf) + Wu*(Uf'*Uf) + ...
             Wdu*(DUf'*DUf) + lambda_g*eye(ng));
    Hqp = (Hqp + Hqp')/2;
    Aineq = [Uf; -Uf; DUf; -DUf];

    qp_options = optimoptions('quadprog', ...
        'Algorithm','interior-point-convex', 'Display','off');

    u_ini = zeros(Tini,1);
    y_ini = zeros(Tini,1);
    u_previous = 0;
    x_true = zeros(nx,1);
    y = zeros(Nsim,1);
    u = zeros(Nsim,1);
    exitflags = zeros(Nsim,1);

    for k = 1:Nsim
        idx = k:min(k+Np-1,Nsim);
        r_horizon = [reference(idx); ...
                     repmat(reference(end),Np-numel(idx),1)];
        d_previous = [-u_previous; zeros(Np-1,1)];

        fqp = -2*Wy*(Yf'*r_horizon) + ...
                2*Wdu*(DUf'*d_previous);

        bineq = [ UMAX*ones(Np,1); ...
                  UMAX*ones(Np,1); ...
                  DUMAX*ones(Np,1)-d_previous; ...
                  DUMAX*ones(Np,1)+d_previous ];

        beq_full = [u_ini; y_ini];
        beq = beq_full(independent_rows);

        [z_star, ~, exitflags(k)] = quadprog( ...
            Hqp, fqp, Aineq, bineq, Aeq, beq, ...
            [], [], [], qp_options);

        if exitflags(k) <= 0
            error(['SVD-DeePC QP failed at sample %d (exit flag %d). ' ...
                   'This strict standalone formulation has no slack; ' ...
                   'inspect the full-trajectory ILC-data rank and conditioning.'], ...
                  k, exitflags(k));
        end

        u_plan = Uf*z_star;
        u(k) = u_plan(1);  % apply only the first optimized input

        % True model appears only here, as the simulated physical system.
        y(k) = Ct*x_true + Dt*u(k);
        x_true = At*x_true + Bt*u(k);

        u_ini = [u_ini(2:end); u(k)];
        y_ini = [y_ini(2:end); y(k)];
        u_previous = u(k);
    end
end

%% Scalar block-Hankel matrix
function H = block_hankel(signal, depth)
    signal = signal(:);
    number_of_columns = numel(signal)-depth+1;
    if number_of_columns < 1
        error('Signal length must be at least the Hankel depth.');
    end

    H = zeros(depth,number_of_columns);
    for row = 1:depth
        H(row,:) = signal(row:row+number_of_columns-1).';
    end
end

%% FIR inverse identification using the PD-type L0
% The same fixed kp/kd/Q-filter update from the original script tracks the
% filtered impulse. The final learned impulse
% input divided by alpha becomes the reusable FIR inverse.
function [f_final, u_final, y_final, relerr_curve] = ...
    identify_inverse_fir_pd(plant, r_inverse, alpha, ...
        kp, kd, Qfilt, t, Ntrials, nx, add_noise, noise_std)

    N = numel(t);
    relerr_curve = zeros(Ntrials,1);
    reference_norm = max(norm(r_inverse), eps);

    % Preserve the original PD-ILC initialization, start the impulse-learning
    % experiment from zero input and learn its required input trial by trial.
    u = zeros(N,1);

    for j = 1:Ntrials
        y_true = lsim(plant, u, t, zeros(nx,1)); % plant IC reset each trial
        y_learn = y_true;
        if add_noise
            y_learn = y_learn + noise_std*randn(size(y_learn));
        end

        e = r_inverse - y_learn;
        relerr_curve(j) = norm(e)/reference_norm;

     
        if j < Ntrials
            e_next = [e(2:end); 0];
            de     = [diff(e); 0];
            u      = filtfilt(Qfilt, u + kp*e_next + kd*de); % PD learning
        end
    end

    u_final = u;
    y_final = lsim(plant, u_final, t, zeros(nx,1));
    f_final = u_final / alpha;
end

%% Inverse-filter ILC for an arbitrary reference
% The learned inverse is reused as a filter, not replayed as the same input.
% u0 = F*r gives the feedforward initialization; later trials apply
% F*e corrections. f_final remains fixed 
% u0 =F*ref
% uj+1 = uj + F* ej
function [u_final, y_final, rmse_curve] = run_inverse_ilc( ...
    plant, r, t, act, f_final, zero_lag_index, Ntrials, nx)

    u = apply_centered_fir(f_final, r, zero_lag_index);
    rmse_curve = zeros(Ntrials,1);

    for j = 1:Ntrials
        y = lsim(plant, u, t, zeros(nx,1));      % plant IC reset each trial
        e = r - y;
        rmse_curve(j) = sqrt(mean(e(act).^2));

        if j < Ntrials
            u = u + apply_centered_fir(f_final, e, zero_lag_index);
        end
    end

    u_final = u;
    y_final = y;
end

%% Apply a possibly noncausal FIR whose zero-lag coefficient is f(k0)
function y = apply_centered_fir(f, x, k0)
    N = numel(x);
    y_full = conv(x, f, 'full'); % convolution x * f
    y = y_full(k0 : k0+N-1); %Filtered signal with the correct noncausal alignment
end

function s = tern(cond, a, b)
    if cond, s = a; else, s = b; end
end
