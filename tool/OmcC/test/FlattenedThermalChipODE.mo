class ThermalChipODE
  final parameter Integer N = 10 "Number of volumes in the x direction";
  final parameter Integer M = 10 "Number of volumes in the y direction";
  final parameter Integer P = 4 "Number of volumes in the z direction";
  final parameter Real L(unit = "m") = 0.012 "Chip length in the x direction";
  final parameter Real W(unit = "m") = 0.012 "Chip width in the y direction";
  final parameter Real H(unit = "m") = 0.004 "Chip height in the z direction";
  final parameter Real lambda(unit = "W/(m.K)") = 148.0 "Thermal conductivity of silicon";
  final parameter Real rho(unit = "kg/m3") = 2329.0 "Density of silicon";
  final parameter Real c(unit = "J/(kg.K)") = 700.0 "Specific heat capacity of silicon";
  parameter Real Tstart(unit = "K", nominal = 500.0) = 313.15;
  final parameter Real l(unit = "m") = 0.0012 "Chip length in the x direction";
  final parameter Real w(unit = "m") = 0.0012 "Chip width in the y direction";
  final parameter Real h(unit = "m") = 0.001 "Chip height in the z direction";
  final parameter Real Tt(unit = "K", nominal = 500.0) = 313.15 "Prescribed temperature of the top surface";
  final parameter Real C(unit = "J/K") = 1630300.0 * 0.0012 * 0.0012 * 0.001 "Thermal capacitance of a volume";
  final parameter Real Gx(unit = "W/K") = 148.0 * 0.0012 * 0.001 / 0.0012 "Thermal conductance of a volume,x direction";
  final parameter Real Gy(unit = "W/K") = 148.0 * 0.0012 * 0.001 / 0.0012 "Thermal conductance of a volume,y direction";
  final parameter Real Gz(unit = "W/K") = 148.0 * 0.0012 * 0.0012 / 0.001 "Thermal conductance of a volume,z direction";
  Real[10, 10, 4] T(unit = "K", start = 313.15, fixed = true, nominal = 500.0) "Temperatures of the volumes";
  Real[10, 10] Qb(unit = "W") "Power injected in the bottom volumes";
  parameter Real Ptot(unit = "W") = 100.0 "Total power consumption";
  final parameter Real Pv(unit = "W") = 100.0 / 50.0 "Power dissipated in a single volume";
equation
for i in 1:10 loop
	for k in 1:5 loop
  Qb[i,k] = Pv;
	end for;
end for;
for i in 1:10 loop
	for k in 6:10 loop
  Qb[i,k] = 0.0;
	end for;
end for;
  der(T[1,1,1]) = 1.0 / C * (Gx * ((-T[1,1,1]) + T[2,1,1]) + Gy * ((-T[1,1,1]) + T[1,2,1]) + Gz * (626.3 - 3.0 * T[1,1,1] + T[1,1,2])) "Upper left top corner";
  der(T[10,1,1]) = 1.0 / C * (Gx * (T[9,1,1] - T[10,1,1]) + Gy * ((-T[10,1,1]) + T[10,2,1]) + Gz * (626.3 - 3.0 * T[10,1,1] + T[10,1,2])) "Lower left top corner";
  der(T[1,10,1]) = 1.0 / C * (Gx * ((-T[1,10,1]) + T[2,10,1]) + Gy * (T[1,9,1] - T[1,10,1]) + Gz * (626.3 - 3.0 * T[1,10,1] + T[1,10,2])) "Upper right top corner";
  der(T[10,10,1]) = 1.0 / C * (Gx * (T[9,10,1] - T[10,10,1]) + Gy * (T[10,9,1] - T[10,10,1]) + Gz * (626.3 - 3.0 * T[10,10,1] + T[10,10,2])) "Lower right top corner";
  der(T[1,1,4]) = 1.0 / C * (Gx * ((-T[1,1,4]) + T[2,1,4]) + Gy * ((-T[1,1,4]) + T[1,2,4]) + Gz * (T[1,1,3] - T[1,1,4]) + Qb[1,1]) "Upper left bottom corner";
  der(T[10,1,4]) = 1.0 / C * (Gx * (T[9,1,4] - T[10,1,4]) + Gy * ((-T[10,1,4]) + T[10,2,4]) + Gz * (T[10,1,3] - T[10,1,4]) + Qb[10,1]) "Lower left bottom corner";
  der(T[1,10,4]) = 1.0 / C * (Gx * ((-T[1,10,4]) + T[2,10,4]) + Gy * (T[1,9,4] - T[1,10,4]) + Gz * (T[1,10,3] - T[1,10,4]) + Qb[1,10]) "Upper right bottom corner";
  der(T[10,10,4]) = 1.0 / C * (Gx * (T[9,10,1] - T[10,10,4]) + Gy * (T[10,9,1] - T[10,10,4]) + Gz * (T[10,10,3] - T[10,10,4]) + Qb[10,10]) "Lower right bottom corner";
  for i in 2:9 loop
    der(T[i,10,4]) = 1.0 / C * (Gx * (T[i - 1,10,4] - 2.0 * T[i,10,4] + T[i + 1,10,4]) + Gy * (T[i,10 - 1,4] - T[i,10,4]) + Gz * (T[i,10,4 - 1] - T[i,10,4]) + Qb[i,10]) "Right bottom edge";
    der(T[i,1,4]) = 1.0 / C * (Gx * (T[i - 1,1,4] - 2.0 * T[i,1,4] + T[i + 1,1,4]) + Gy * ((-T[i,1,4]) + T[i,2,4]) + Gz * (T[i,1,4 - 1] - T[i,1,4]) + Qb[i,1]) "Left bottom edge";
    der(T[i,10,1]) = 1.0 / C * (Gx * (T[i - 1,10,1] - 2.0 * T[i,10,1] + T[i + 1,10,1]) + Gy * (T[i,10 - 1,1] - T[i,10,1]) + Gz * (2.0 * 313.15 - 3.0 * T[i,10,1] + T[i,10,2])) "Right top edge";
    der(T[i,1,1]) = 1.0 / C * (Gx * (T[i - 1,1,1] - 2.0 * T[i,1,1] + T[i + 1,1,1]) + Gy * ((-T[i,1,1]) + T[i,2,1]) + Gz * (2.0 * 313.15 - 3.0 * T[i,1,1] + T[i,1,2])) "Left top edge";
  end for;
  for j in 2:9 loop
    der(T[10,j,4]) = 1.0 / C * (Gx * (T[10,j - 1,4] - 2.0 * T[10,j,4] + T[10,j + 1,4]) + Gy * (T[10 - 1,j,4] - T[10,j,4]) + Gz * (T[10,j,4 - 1] - T[10,j,4]) + Qb[10,j]) "Lower bottom edge";
    der(T[1,j,4]) = 1.0 / C * (Gx * (T[1,j - 1,4] - 2.0 * T[1,j,4] + T[1,j + 1,4]) + Gy * ((-T[1,j,4]) + T[2,j,4]) + Gz * (T[1,j,4 - 1] - T[1,j,4]) + Qb[1,j]) "Upper bottom edge";
    der(T[10,j,1]) = 1.0 / C * (Gx * (T[10,j - 1,1] - 2.0 * T[10,j,1] + T[10,j + 1,1]) + Gy * (T[10 - 1,j,1] - T[10,j,1]) + Gz * (2.0 * 313.15 - 3.0 * T[10,j,1] + T[10,j,2])) "Lower top edge";
    der(T[1,j,1]) = 1.0 / C * (Gx * (T[1,j - 1,1] - 2.0 * T[1,j,1] + T[1,j + 1,1]) + Gy * ((-T[1,j,1]) + T[2,j,1]) + Gz * (2.0 * 313.15 - 3.0 * T[1,j,1] + T[1,j,2])) "Upper top edge";
  end for;
  for k in 2:3 loop
    der(T[10,10,k]) = 1.0 / C * (Gx * (T[10 - 1,10,k] - T[10,10,k]) + Gy * (T[10,10 - 1,k] - T[10,10,k]) + Gz * (T[10,10,k - 1] - 2.0 * T[10,10,k] + T[10,10,k + 1])) "Lower right edge";
    der(T[1,10,k]) = 1.0 / C * (Gx * (T[1,10 - 1,k] - T[1,10,k]) + Gy * (T[2,10,k] - T[1,10,k]) + Gz * (T[1,10,k - 1] - 2.0 * T[1,10,k] + T[1,10,k + 1])) "Upper right edge";
    der(T[10,1,k]) = 1.0 / C * (Gx * (T[10 - 1,1,k] - T[10,1,k]) + Gy * ((-T[10,1,k]) + T[10,2,k]) + Gz * (T[10,1,k - 1] - 2.0 * T[10,1,k] + T[10,1,k + 1])) "Lower left edge";
    der(T[1,1,k]) = 1.0 / C * (Gx * ((-T[1,1,k]) + T[2,1,k]) + Gy * ((-T[1,1,k]) + T[1,2,k]) + Gz * (T[1,1,k - 1] - 2.0 * T[1,1,k] + T[1,1,k + 1])) "Upper left edge";
  end for;
  for i in 2:9 loop
    for j in 2:9 loop
      der(T[i,j,4]) = 1.0 / C * (Gx * (T[i - 1,j,4] - 2.0 * T[i,j,4] + T[i + 1,j,4]) + Gy * (T[i,j - 1,4] - 2.0 * T[i,j,4] + T[i,j + 1,4]) + Gz * (T[i,j,4 - 1] - T[i,j,4]) + Qb[i,j]) "Bottom face";
      der(T[i,j,1]) = 1.0 / C * (Gx * (T[i - 1,j,1] - 2.0 * T[i,j,1] + T[i + 1,j,1]) + Gy * (T[i,j - 1,1] - 2.0 * T[i,j,1] + T[i,j + 1,1]) + Gz * (2.0 * 313.15 - 3.0 * T[i,j,1] + T[1,j,2])) "Top face";
    end for;
  end for;
  for i in 2:9 loop
    for k in 2:3 loop
      der(T[i,10,k]) = 1.0 / C * (Gx * (T[i - 1,10,k] - 2.0 * T[i,10,k] + T[i + 1,10,k]) + Gy * (T[i,10 - 1,k] - T[i,10,k]) + Gz * (T[i,10,k - 1] - 2.0 * T[i,10,k] + T[i,10,k + 1])) "Right face";
      der(T[i,1,k]) = 1.0 / C * (Gx * (T[i - 1,1,k] - 2.0 * T[i,1,k] + T[i + 1,1,k]) + Gy * ((-T[i,1,k]) + T[i,2,k]) + Gz * (T[i,1,k - 1] - 2.0 * T[i,1,k] + T[i,1,k + 1])) "Left face";
    end for;
  end for;
  for j in 2:9 loop
    for k in 2:3 loop
      der(T[10,j,k]) = 1.0 / C * (Gx * (T[10 - 1,j,k] - T[10,j,k]) + Gy * (T[10,j - 1,k] - 2.0 * T[10,j,k] + T[10,j + 1,k]) + Gz * (T[10,j,k - 1] - 2.0 * T[10,j,k] + T[10,j,k + 1])) "Lower face";
      der(T[1,j,k]) = 1.0 / C * (Gx * ((-T[1,j,k]) + T[2,j,k]) + Gy * (T[1,j - 1,k] - 2.0 * T[1,j,k] + T[1,j + 1,k]) + Gz * (T[1,j,k - 1] - 2.0 * T[1,j,k] + T[1,j,k + 1])) "Upper face";
    end for;
  end for;
  for i in 2:9 loop
    for j in 2:9 loop
      for k in 2:3 loop
        der(T[i,j,k]) = 1.0 / C * (Gx * (T[i - 1,j,k] - 2.0 * T[i,j,k] + T[i + 1,j,k]) + Gy * (T[i,j - 1,k] - 2.0 * T[i,j,k] + T[i,j + 1,k]) + Gz * (T[i,j,k - 1] - 2.0 * T[i,j,k] + T[i,j,k + 1])) "Internal volume";
      end for;
    end for;
  end for;
end ThermalChipODE;
