model SimpleFirstOrder
    Real x(start = 0, fixed = true);
equation
    der(x) = 1 - x;
end SimpleFirstOrder;