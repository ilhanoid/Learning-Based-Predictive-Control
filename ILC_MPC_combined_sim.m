% Main Idea: ILC learns accurately but ignores constraints; MPC respects
% constraints but needs a model. ILC supplies the data for the model,
% whereas MPC supplies a reference ILC can actually achieve within the limits.

% Combined ILC + MPC scheme:
%     -> Learn one fixed FIR inverse from filtered-impulse tracking
%     -> Reuse that inverse for triangular-reference ILC on the true plant
%     -> Collect u_ILC and y_ILC
%     -> Identify the system model G_hat from the input-output data (System ID Toolbox)
%     -> Use G_hat inside MPC, with input and input-rate constraints
%     -> MPC output becomes the feasible(attainable) version of the reference
%     -> Use that feasible reference for ILC
%     -> Collect new data, update G_hat
%     -> Repeat


%% Implementation Details
%
% ITERATIVE LEARNING CONTROL (ILC) -
%   • A filtered impulse is first tracked on the true plant using the original
%     fixed PD-type learning law as L0.
%   • After this identification phase converges, the learned impulse input
%     is divided by its scale alpha and stored as the FIR inverse f_final.
%   • f_final is then reused for every triangular / MPC-shaped reference:
%         u0      = f_final * r
%         u_{j+1} = u_j + f_final * e_j
%    * denotes centered, generally noncausal FIR convolution.
%   • f_final is fixed after impulse identification; it is not updated inside later trials.
%   • Initial plant states are reset before every trial.
%   • Performance metrics are evaluated over the unpadded region only.
%
% SYSTEM IDENTIFICATION
%   • Implemented using the MATLAB System Identification Toolbox.
%   • Each outer iteration's input-output data (u, y) is appended as a separate experiment using merge.
%   • A second-order discrete-time state-space model with no direct
%     feedthrough (D = 0) is estimated using ssest at sample time Ts.
%   • rng(1) fixes the noise sequence for reproducibility.
%
% MODEL PREDICTIVE CONTROL (MPC)
%   • Implemented using an MPC Toolbox mpc object and simulated with sim.
%   • sim performs receding-horizon closed-loop control automatically.
%   • Reference preview is disabled:
%         RefLookAhead = off
%   • Hard constraints are imposed on the input and input rate.
%   • Constraints are tightened using a back-off factor to accommodate
%     residual model-estimation error.
%
% EXPERIMENTAL TOGGLES
%   FIRST_PASS_SCALE = 1          Disable initial reference scaling.
%   ADD_ID_NOISE = false          Disable forward-ID measurement noise.
%   ADD_INVERSE_ID_NOISE = false  Disable noise during inverse identification.                      
%   G_hat = true_plant            Bypass identification part and use the true model for Mpc
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

% Independent, deterministic validation input used only to compare the
% identified model G_hat with the known true plant G. It is not included in
% the identification dataset. Several frequencies plus a DC component make
% this more informative than validating on another copy of the ILC input.
u_val = zeros(N,1);
u_val(act) = 0.20 ...
           + 0.14*sin(2*pi*0.35*t_active) ...
           + 0.10*sin(2*pi*0.90*t_active) ...
           + 0.06*sin(2*pi*1.80*t_active);
y_val_true = lsim(true_plant, u_val, t, zeros(nx,1));

%% ==================== 3. Constraints & Noise ====================
% Chosen such that the input is bounded
UMAX  = 0.6;        % |u|  <= UMAX
DUMAX = 0.010;      % |du| <= DUMAX per step
ADD_ID_NOISE = true;  % this noise added to the y_ilc before it is used in the sys. identification block
noise_std = 0.010;   % try 0.001, 0.002, 0.005
rng(1);  % generating the same random noise sequence every run

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
% the initial data collection for system identification stays within input
% limits. Set FIRST_PASS_SCALE = 1.0 to disable scaling and use the full
% original reference from the beginning.
FIRST_PASS_SCALE = 1;

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

%% ==================== 6. MPC settings ====================
Np = 80;   Nc = 20;     % horizons (settling ~75 samples)
Wy = 20;   Wu = 0.1;   Wdu = 0.1;

%% ==================== 7. OUTER LOOP ====================
NouterMax = 5;

% Initial ILC target.
% If FIRST_PASS_SCALE < 1, ILC first tracks a scaled-down reference.

r_ilc = FIRST_PASS_SCALE * r_orig;

fprintf('Initial ILC reference scale = %.2f (1.00 = no scaling)\n', FIRST_PASS_SCALE);

allData = [];                    % accumulated identification dataset
hist = struct('rmse_vs_ref',[], 'rmse_vs_orig',[], 'maxu',[], 'maxdu',[], ...
             'fit_true',[], 'viol_u',[], 'viol_du',[]);
R_hist = zeros(N, NouterMax);     % stores the reference trajectory used by ILC at each outer iteration (feasible reference)
Y_hist = zeros(N, NouterMax);     % stores the ILC output at each outer iteration
U_hist = zeros(N, NouterMax);     % stores the final learned ILC input at each outer iteration
Umpc_hist  = zeros(N, NouterMax); % stores the MPC-optimized input at each outer iteration
Yfeas_hist = zeros(N, NouterMax); % stores the MPC output (= feasible reference) at each outer iteration

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

    % fprintf('  ILC : RMSE vs current ref = %.4f | RMSE vs ORIGINAL = %.4f\n', sqrt(mean(e_ref(act).^2)), sqrt(mean(e_orig(act).^2)));
    % fprintf('        max|u| = %.4f %s | max|du| = %.5f %s\n', ...
    %         maxu,  tern(maxu  > UMAX,  '(VIOLATES)', '(ok)'), ...
    %         maxdu, tern(maxdu > DUMAX, '(VIOLATES)', '(ok)'));

    % ---------- 7b. System identification from the ILC data ----------
    % System identification uses the input-output data produced while ILC tracks the current reference. 
    % Accumulate every outer iteration as a separate experiment. Multiple trajectories widen the excitation compared with using only one final
    % converged ILC trajectory.
  
    y_id = y_ilc;   % Adding zero-mean Gaussian measurement noise to the output used for system identification 
                    % This tests robustness of the identified model, not noisy ILC learning
    
    if ADD_ID_NOISE
        y_id = y_id + noise_std * randn(size(y_id));
    end

    dnew = iddata(y_id, u_ilc, Ts); % creates a data object for system identification
    if isempty(allData)
        allData = dnew;
    else
        allData = merge(allData, dnew);  % stores the new data as another experiment inside the same iddata object 
    end                                  % allData={(u 1 ​ ,y 1 ​ ),(u 2 ​ ,y 2 ​ ),...,(u n ​ ,y n ​ )}

    nx_id  = 2;       % model order (true plant is 2nd order)
    opt    = ssestOptions('EnforceStability', true);
    sys_id = ssest(allData, nx_id, opt, 'Feedthrough', false);   % ssest()estimates one model using all collected ILC datasets.
                                                                 % Feedthrough=false sets D=0, so the input does not affect the output instantly at the same sample
    
   
    G_hat = ss(sys_id);
    G_hat = minreal(G_hat);                     % Remove unnecessary/cancelled dynamics (ie. pole-zero cancellations or redundant states)

    
    %  G_hat vs true_plant--> 100*(1 - ||y_true-y_hat|| / ||y_true-mean(y_true)||
    y_val_hat = lsim(G_hat, u_val, t, zeros(size(G_hat.A,1),1));
    val_denom = norm(y_val_true(act) - mean(y_val_true(act)));
    fit_true = 100*(1 - norm(y_val_true(act) - y_val_hat(act))/val_denom);

    fprintf('  ID  : true-plant validation fit = %.2f%%\n', fit_true);

    % ---------- 7c. MPC: Generate the feasible reference ----------
    % MPC always tracks the original reference. Because it must respect the
    % constraints, its achievable output is the attainable version of that
    % trajectory, which becomes the next ILC target.
    mpcobj = mpc(G_hat, Ts, Np, Nc);          % creates an MPC controller object.
    mpcobj.MV.Min     = -UMAX;
    mpcobj.MV.Max     =  UMAX;
    mpcobj.MV.RateMin = -DUMAX;
    mpcobj.MV.RateMax =  DUMAX;
    mpcobj.Weights.OutputVariables          = Wy;
    mpcobj.Weights.ManipulatedVariables     = Wu;
    mpcobj.Weights.ManipulatedVariablesRate = Wdu;
    simopt = mpcsimopt(mpcobj);              % specifies additional simulation options object for the MPC simulation
    simopt.RefLookAhead = 'on';

    [y_feasible, ~, u_mpc] = sim(mpcobj, N, r_orig, simopt);  % runs the closed-loop MPC simulation, ~ ignores the time vector

    fprintf('  MPC : feasible-ref RMSE vs ORIGINAL = %.4f | max|u| = %.4f\n', ...
            sqrt(mean((r_orig(act)-y_feasible(act)).^2)), max(abs(u_mpc(act))));

    % ---------- 7d. logging ----------
    hist.rmse_vs_ref(outer)  = sqrt(mean(e_ref(act).^2));
    hist.rmse_vs_orig(outer) = sqrt(mean(e_orig(act).^2));
    hist.maxu(outer)         = maxu;
    hist.maxdu(outer)        = maxdu;
    hist.fit_true(outer)     = fit_true;
    hist.viol_u(outer)       = maxu  > UMAX;
    hist.viol_du(outer)      = maxdu > DUMAX;

    R_hist(:,outer)     = r_ilc;
    Y_hist(:,outer)     = y_ilc;
    U_hist(:,outer)     = u_ilc;
    Umpc_hist(:,outer)  = u_mpc;
    Yfeas_hist(:,outer) = y_feasible;

    % ---------- 7e. feasible reference becomes the next ILC target ----------
    r_ilc = y_feasible;
    fprintf('\n');
end

%% ==================== 8. Summary ====================
T = table((1:NouterMax)', hist.rmse_vs_ref(:), hist.rmse_vs_orig(:), ...
          hist.maxu(:), hist.maxdu(:), hist.fit_true(:), ...
          'VariableNames', {'Outer','RMSE_vs_ref','RMSE_vs_orig','MaxU','MaxdU', ...
                            'TruePlantFit'});
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
plot(t, R_hist(:,2), 'b-',  'LineWidth',1.3);
plot(t, R_hist(:,NouterMax), 'r-', 'LineWidth',1.3);
xlabel('Time [s]'); ylabel('Reference'); grid on;
legend('original (desired)','feasible, iter 2','feasible, final','Location','best');
title('MPC reshapes the reference into an attainable trajectory');

subplot(2,1,2);
stairs(t, U_hist(:,1), 'b-', 'LineWidth',1.2); hold on;
stairs(t, U_hist(:,NouterMax), 'r-', 'LineWidth',1.3);
yline( UMAX,'k--','LineWidth',1.2); yline(-UMAX,'k--','LineWidth',1.2);
xlabel('Time [s]'); ylabel('Input u'); grid on;
legend('ILC input, iter 1','ILC input, final','constraints','Location','best');
title('ILC input before vs. after reference reshaping');

%% ---- MPC vs ILC: inputs and outputs (final outer iteration) ----
% Left column  = MPC stage : tracks the original reference under constraints;
%                its output is the feasible reference handed to ILC.
% Right column = ILC stage : tracks that feasible reference on the true plant.

kF = NouterMax;

figure('Name','MPC vs ILC: inputs and outputs','Position',[150 60 1150 720]);

subplot(2,2,1);
plot(t, r_orig, 'k--', 'LineWidth',1.4); hold on;
plot(t, Yfeas_hist(:,kF), 'b-', 'LineWidth',1.5);
xlabel('Time [s]'); ylabel('Output'); grid on;
legend('original reference','MPC output = feasible ref','Location','best');
title(sprintf('MPC stage - output (iter %d)', kF));

subplot(2,2,2);
stairs(t, Umpc_hist(:,kF), 'b-', 'LineWidth',1.3); hold on;
yline( UMAX,'k--','LineWidth',1.2); yline(-UMAX,'k--','LineWidth',1.2);
yline( UMAX,'k:','LineWidth',1.0); yline(-UMAX,'k:','LineWidth',1.0);
xlabel('Time [s]'); ylabel('u_{MPC}'); grid on;
legend('MPC optimized input','constraint','back-off','Location','best');
title(sprintf('MPC stage - input   (max|u| = %.4f)', max(abs(Umpc_hist(act,kF)))));

subplot(2,2,3);
plot(t, R_hist(:,kF), 'k--', 'LineWidth',1.4); hold on;
plot(t, Y_hist(:,kF), 'r-', 'LineWidth',1.5);
plot(t, r_orig, 'Color',[.6 .6 .6], 'LineStyle',':', 'LineWidth',1.1);
xlabel('Time [s]'); ylabel('Output'); grid on;
legend('feasible ref (ILC target)','ILC output','original ref','Location','best');
title(sprintf('ILC stage - output (iter %d)', kF));

subplot(2,2,4);
stairs(t, U_hist(:,kF), 'r-', 'LineWidth',1.3); hold on;
yline( UMAX,'k--','LineWidth',1.2); yline(-UMAX,'k--','LineWidth',1.2);
xlabel('Time [s]'); ylabel('u_{ILC}'); grid on;
legend('ILC learned input','constraint','Location','best');
title(sprintf('ILC stage - input   (max|u| = %.4f %s)', ...
      max(abs(U_hist(act,kF))), tern(max(abs(U_hist(act,kF)))>UMAX,'VIOLATES','ok')));

sgtitle('MPC generates the feasible reference; ILC then learns to track it on the true plant');

%% ==================== local functions ====================
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
