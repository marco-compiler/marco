model SimpleThermalDAE
  Real[4] T(start = 100);
  Real[5] Q;
equation
  for i in 1:4 loop
    der(T[i]) = Q[i] - Q[i + 1];
  end for;
  Q[1] = 10;
  for i in 2:4 loop
    Q[i] = T[i - 1] - T[i];
  end for;
  Q[5] = 10;
end SimpleThermalDAE;
