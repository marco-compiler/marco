model Loop22
  Real x(start = -2);
  Real y(start = -2);
equation
  x * x + y * y + x + y = 0;
  x * x * y + x = y;
end Loop22;
