
    model HarmonicOscillator
      parameter Real m = 1;
      parameter Real k = 10; 
      Real[4] x(start = 1.0);
      Real[4] v(start = 0.0);
    equation
      for i in 1:4 loop
        der(x[i]) = v[i];
      end for;
      m*der(v[1]) = k*(x[2]-x[1]);
      for i in 2:3 loop
        m*der(v[i]) = k*(x[i-1] - x[i]) + k*(x[i+1] - x[i]);
      end for;
      m*der(v[4]) = k*(x[3]-x[4]);
    end HarmonicOscillator;
