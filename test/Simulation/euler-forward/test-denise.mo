model rc
  Real a[1000], b[1000], x[1000] (each fixed = true, each start = 0);
equation
  for i in 2:999 loop
    der(x[i]) = a[i] - x[i];
    a[i + 1] = a[i] + b[i];
    b[i] = x[i - 1];
  end for;
  b[1] = 0;
  b[1000] = x[999];
  der(x[1]) = a[1] - x[1];
  der(x[1000]) = a[1000] - x[1000];
  a[1] = 1;
  a[1000] = a[999] + b[999];
end rc;