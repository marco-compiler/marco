    model CascadedFirstOrder
      parameter Real T = 1 "System delay";
      final parameter Real tau = T/10 "Individual time constant";
      Real[10] x(start = 0);
    equation
      tau*der(x[1]) = 1 - x[1];
      for i in 2:10 loop
        tau*der(x[i]) = x[i-1] - x[i];
      end for;
    end CascadedFirstOrder;
  
