function p = System_Inputs(t)
    % System_Inputs: delta_fl, delta_fr, delta_rl, delta_rr and Fw
    p = [1.0; ... % delta_fl (control input)
         1.0; ... % delta_fr (control input)
         1.0; ... % delta_rl (control input)
         1.0; ... % delta_rr (control input)
         1.0];    % Fw (disturbance input)
end