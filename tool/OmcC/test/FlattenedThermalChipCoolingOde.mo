class ThermalChipCoolingODE
  final parameter Integer N = 4 "Number of volumes in the x direction";
  final parameter Integer M = 4 "Number of volumes in the y direction";
  final parameter Integer P = 4 "Number of volumes in the z direction";
  final parameter Real L(unit = "m") = 0.012 "Chip length in the x direction";
  final parameter Real W(unit = "m") = 0.012 "Chip width in the y direction";
  final parameter Real H(unit = "m") = 0.004 "Chip height in the z direction";
  final parameter Real S(unit = "m") = 0.001 "Cooling channel thickness";
  final parameter Real lambda(unit = "W/(m.K)") = 148.0 "Thermal conductivity of silicon";
  final parameter Real rho(unit = "kg/m3") = 2329.0 "Density of silicon";
  final parameter Real c(unit = "J/(kg.K)") = 700.0 "Specific heat capacity of silicon";
  parameter Real rhof(unit = "kg/m3") = 1000.0 "Density of the cooling fluid";
  parameter Real cf(unit = "J/(kg.K)") = 4200.0 "Specific heat capacity of the cooling fluid";
  parameter Real Tstart(unit = "K", nominal = 500.0) = 298.15 "Start value of temperature";
  parameter Real Tc(unit = "K", nominal = 500.0) = 298.15 "Inlet temperature of the cooling fluid";
  parameter Real gamma_n(unit = "W/(m2.K)") = 50000.0 "Nominal heat transfer coefficient";
  parameter Real alpha(unit = "1") = 0.8 "Exponent of mass flow rate in heat transfer correlation";
  parameter Real wn(unit = "kg/s") = 0.001 "Nominal coolant mass flow rate";
  final parameter Real l(unit = "m") = 0.003 "Chip length in the x direction";
  final parameter Real w(unit = "m") = 0.003 "Chip width in the y direction";
  final parameter Real h(unit = "m") = 0.001 "Chip height in the z direction";
  final parameter Real C(unit = "J/K") = 1630300.0 * l * w * h "Thermal capacitance of a volume";
  final parameter Real Gx(unit = "W/K") = 148.0 * w * h / l "Thermal conductance of a volume,x direction";
  final parameter Real Gy(unit = "W/K") = 148.0 * l * h / w "Thermal conductance of a volume,y direction";
  final parameter Real Gz(unit = "W/K") = 148.0 * l * w / h "Thermal conductance of a volume,z direction";
  Real[4, 4, 4] T(unit = "K", start = Tstart, fixed = true, nominal = 500.0) "Temperatures of the volumes";
  Real[4, 4] Qb(unit = "W") "Power injected in the bottom volumes";
  Real[4, 4] Qf(unit = "W") "Thermal power entering each fluid volume";
  Real[4, 4] Tf(unit = "K", start = Tstart, fixed = true, nominal = 500.0) "Outlet temperature of each fluid volume";
  Real[4, 4] Ta(unit = "K", nominal = 500.0) "Average temperature of each fluid volume";
  Real[4] gamma(unit = "W/(m2.K)") "Coefficient of heat transfer";
  parameter Real wtot(unit = "kg/s") = 0.001;
  Real[4] wc(unit = "kg/s") "Mass flow rate through each channel";
  parameter Real Ptot(unit = "W") = 100.0 "Total power consumption";
  final parameter Real Pv(unit = "W") = Ptot / 8.0 "Power dissipated in a single volume";
equation
  for j in 1:4 loop
    for i in 1:2 loop
      Qb[j,i] = Pv;
    end for;
  end for;
  for j in 1:4 loop
    for i in 2 + 1:4 loop
      Qb[j,i] = 0.0;
    end for;
  end for;
  for j in 1:4 loop
    wc[j] = wtot * (1.0 + 0.1 * (/*Real*/(j - 1) - /*Real*/(4 - 1) / 2.0) / /*Real*/(4)) / /*Real*/(4);
  end for;
  for j in 1:4 loop
    for i in 2:4 loop
      der(Tf[i,j]) = 1.0 / (rhof * w * 0.001 * l * cf) * (wc[j] * cf * (Tf[i - 1,j] - Tf[i,j]) + Qf[i,j]);
    end for;
    der(Tf[1,j]) = 1.0 / (rhof * w * 0.001 * l * cf) * (wc[j] * cf * (Tc - Tf[1,j]) + Qf[1,j]);
    for i in 1:4 loop
      Qf[i,j] = 2.0 * Gz * gamma[j] * w * l / (2.0 * Gz + gamma[j] * w * l) * (T[i,j,1] - Ta[i,j]);
    end for;
    for i in 2:4 loop
      Ta[i,j] = (Tf[i - 1,j] + Tf[i,j]) / 2.0;
    end for;
    Ta[1,j] = (Tc + Tf[1,j]) / 2.0;
    gamma[j] = (wc[j] / (wn / /*Real*/(4))) ^ alpha * gamma_n;
  end for;
  der(T[1,1,1]) = 1.0 / C * (Gx * ((-T[1,1,1]) + T[2,1,1]) + Gy * ((-T[1,1,1]) + T[1,2,1]) + Gz * ((-T[1,1,1]) + T[1,1,2]) - Qf[1,1]) "Upper left top corner";
  der(T[4,1,1]) = 1.0 / C * (Gx * (T[3,1,1] - T[4,1,1]) + Gy * ((-T[4,1,1]) + T[4,2,1]) + Gz * ((-T[4,1,1]) + T[4,1,2]) - Qf[4,1]) "Lower left top corner";
  der(T[1,4,1]) = 1.0 / C * (Gx * ((-T[1,4,1]) + T[2,4,1]) + Gy * (T[1,3,1] - T[1,4,1]) + Gz * ((-T[1,4,1]) + T[1,4,2]) - Qf[1,4]) "Upper right top corner";
  der(T[4,4,1]) = 1.0 / C * (Gx * (T[3,4,1] - T[4,4,1]) + Gy * (T[4,3,1] - T[4,4,1]) + Gz * ((-T[4,4,1]) + T[4,4,2]) - Qf[4,4]) "Lower right top corner";
  der(T[1,1,4]) = 1.0 / C * (Gx * ((-T[1,1,4]) + T[2,1,4]) + Gy * ((-T[1,1,4]) + T[1,2,4]) + Gz * (T[1,1,3] - T[1,1,4]) + Qb[1,1]) "Upper left bottom corner";
  der(T[4,1,4]) = 1.0 / C * (Gx * (T[3,1,4] - T[4,1,4]) + Gy * ((-T[4,1,4]) + T[4,2,4]) + Gz * (T[4,1,3] - T[4,1,4]) + Qb[4,1]) "Lower left bottom corner";
  der(T[1,4,4]) = 1.0 / C * (Gx * ((-T[1,4,4]) + T[2,4,4]) + Gy * (T[1,3,4] - T[1,4,4]) + Gz * (T[1,4,3] - T[1,4,4]) + Qb[1,4]) "Upper right bottom corner";
  der(T[4,4,4]) = 1.0 / C * (Gx * (T[3,4,4] - T[4,4,4]) + Gy * (T[4,3,4] - T[4,4,4]) + Gz * (T[4,4,3] - T[4,4,4]) + Qb[4,4]) "Lower right bottom corner";
  for i in 2:4 - 1 loop
    der(T[i,4,4]) = 1.0 / C * (Gx * (T[i - 1,4,4] - 2.0 * T[i,4,4] + T[i + 1,4,4]) + Gy * (T[i,4 - 1,4] - T[i,4,4]) + Gz * (T[i,4,4 - 1] - T[i,4,4]) + Qb[i,4]) "Right bottom edge";
    der(T[i,1,4]) = 1.0 / C * (Gx * (T[i - 1,1,4] - 2.0 * T[i,1,4] + T[i + 1,1,4]) + Gy * ((-T[i,1,4]) + T[i,2,4]) + Gz * (T[i,1,4 - 1] - T[i,1,4]) + Qb[i,1]) "Left bottom edge";
    der(T[i,4,1]) = 1.0 / C * (Gx * (T[i - 1,4,1] - 2.0 * T[i,4,1] + T[i + 1,4,1]) + Gy * (T[i,4 - 1,1] - T[i,4,1]) + Gz * ((-T[i,4,1]) + T[i,4,2]) - Qf[i,4]) "Right top edge";
    der(T[i,1,1]) = 1.0 / C * (Gx * (T[i - 1,1,1] - 2.0 * T[i,1,1] + T[i + 1,1,1]) + Gy * ((-T[i,1,1]) + T[i,2,1]) + Gz * ((-T[i,1,1]) + T[i,1,2]) - Qf[i,1]) "Left top edge";
  end for;
  for j in 2:4 - 1 loop
    der(T[4,j,4]) = 1.0 / C * (Gx * (T[4,j - 1,4] - 2.0 * T[4,j,4] + T[4,j + 1,4]) + Gy * (T[4 - 1,j,4] - T[4,j,4]) + Gz * (T[4,j,4 - 1] - T[4,j,4]) + Qb[4,j]) "Lower bottom edge";
    der(T[1,j,4]) = 1.0 / C * (Gx * (T[1,j - 1,4] - 2.0 * T[1,j,4] + T[1,j + 1,4]) + Gy * ((-T[1,j,4]) + T[2,j,4]) + Gz * (T[1,j,4 - 1] - T[1,j,4]) + Qb[1,j]) "Upper bottom edge";
    der(T[4,j,1]) = 1.0 / C * (Gx * (T[4,j - 1,1] - 2.0 * T[4,j,1] + T[4,j + 1,1]) + Gy * (T[4 - 1,j,1] - T[4,j,1]) + Gz * ((-T[4,j,1]) + T[4,j,2]) - Qf[4,j]) "Lower top edge";
    der(T[1,j,1]) = 1.0 / C * (Gx * (T[1,j - 1,1] - 2.0 * T[1,j,1] + T[1,j + 1,1]) + Gy * ((-T[1,j,1]) + T[2,j,1]) + Gz * ((-T[1,j,1]) + T[1,j,2]) - Qf[1,j]) "Upper top edge";
  end for;
  for k in 2:4 - 1 loop
    der(T[4,4,k]) = 1.0 / C * (Gx * (T[4 - 1,4,k] - T[4,4,k]) + Gy * (T[4,4 - 1,k] - T[4,4,k]) + Gz * (T[4,4,k - 1] - 2.0 * T[4,4,k] + T[4,4,k + 1])) "Lower right edge";
    der(T[1,4,k]) = 1.0 / C * (Gx * (T[1,4 - 1,k] - T[1,4,k]) + Gy * (T[2,4,k] - T[1,4,k]) + Gz * (T[1,4,k - 1] - 2.0 * T[1,4,k] + T[1,4,k + 1])) "Upper right edge";
    der(T[4,1,k]) = 1.0 / C * (Gx * (T[4 - 1,1,k] - T[4,1,k]) + Gy * ((-T[4,1,k]) + T[4,2,k]) + Gz * (T[4,1,k - 1] - 2.0 * T[4,1,k] + T[4,1,k + 1])) "Lower left edge";
    der(T[1,1,k]) = 1.0 / C * (Gx * ((-T[1,1,k]) + T[2,1,k]) + Gy * ((-T[1,1,k]) + T[1,2,k]) + Gz * (T[1,1,k - 1] - 2.0 * T[1,1,k] + T[1,1,k + 1])) "Upper left edge";
  end for;
  for i in 2:4 - 1 loop
    for j in 2:4 - 1 loop
      der(T[i,j,4]) = 1.0 / C * (Gx * (T[i - 1,j,4] - 2.0 * T[i,j,4] + T[i + 1,j,4]) + Gy * (T[i,j - 1,4] - 2.0 * T[i,j,4] + T[i,j + 1,4]) + Gz * (T[i,j,4 - 1] - T[i,j,4]) + Qb[i,j]) "Bottom face";
      der(T[i,j,1]) = 1.0 / C * (Gx * (T[i - 1,j,1] - 2.0 * T[i,j,1] + T[i + 1,j,1]) + Gy * (T[i,j - 1,1] - 2.0 * T[i,j,1] + T[i,j + 1,1]) + Gz * ((-T[i,j,1]) + T[i,j,2]) - Qf[i,j]) "Top face";
    end for;
  end for;
  for i in 2:4 - 1 loop
    for k in 2:4 - 1 loop
      der(T[i,4,k]) = 1.0 / C * (Gx * (T[i - 1,4,k] - 2.0 * T[i,4,k] + T[i + 1,4,k]) + Gy * (T[i,4 - 1,k] - T[i,4,k]) + Gz * (T[i,4,k - 1] - 2.0 * T[i,4,k] + T[i,4,k + 1])) "Right face";
      der(T[i,1,k]) = 1.0 / C * (Gx * (T[i - 1,1,k] - 2.0 * T[i,1,k] + T[i + 1,1,k]) + Gy * ((-T[i,1,k]) + T[i,2,k]) + Gz * (T[i,1,k - 1] - 2.0 * T[i,1,k] + T[i,1,k + 1])) "Left face";
    end for;
  end for;
  for j in 2:4 - 1 loop
    for k in 2:4 - 1 loop
      der(T[4,j,k]) = 1.0 / C * (Gx * (T[4 - 1,j,k] - T[4,j,k]) + Gy * (T[4,j - 1,k] - 2.0 * T[4,j,k] + T[4,j + 1,k]) + Gz * (T[4,j,k - 1] - 2.0 * T[4,j,k] + T[4,j,k + 1])) "Lower face";
      der(T[1,j,k]) = 1.0 / C * (Gx * ((-T[1,j,k]) + T[2,j,k]) + Gy * (T[1,j - 1,k] - 2.0 * T[1,j,k] + T[1,j + 1,k]) + Gz * (T[1,j,k - 1] - 2.0 * T[1,j,k] + T[1,j,k + 1])) "Upper face";
    end for;
  end for;
  for i in 2:4 - 1 loop
    for j in 2:4 - 1 loop
      for k in 2:4 - 1 loop
        der(T[i,j,k]) = 1.0 / C * (Gx * (T[i - 1,j,k] - 2.0 * T[i,j,k] + T[i + 1,j,k]) + Gy * (T[i,j - 1,k] - 2.0 * T[i,j,k] + T[i,j + 1,k]) + Gz * (T[i,j,k - 1] - 2.0 * T[i,j,k] + T[i,j,k + 1])) "Internal volume";
      end for;
    end for;
  end for;
end ThermalChipCoolingODE;

