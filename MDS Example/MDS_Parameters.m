function p = MDS_Parameters(t)
    p = [3.0 + 1.2*exp(-0.3*t)*cos(10*t); ... % m
         2.0 + 0.6*exp(-0.3*t)*sin(10*t); ... % k
         0.8 + 0.01*t];                       % c