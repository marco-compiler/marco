class ThermalChipODE
  final parameter Integer aN = 10 "Number of volumes in the x direction";
  final parameter Integer aM = 10 "Number of volumes in the y direction";
  final parameter Integer aP = 4 "Number of volumes in the z direction";
  final parameter Real aL(unit = "m") = 0.012 "Chip length in the x direction";
  final parameter Real aW(unit = "m") = 0.012 "Chip width in the y direction";
  final parameter Real aH(unit = "m") = 0.004 "Chip height in the z direction";
  final parameter Real alambda(unit = "W/(m.K)") = 148.0 "Thermal conductivity of silicon";
  final parameter Real arho(unit = "kg/m3") = 2329.0 "Density of silicon";
  final parameter Real ac(unit = "J/(kg.K)") = 700.0 "Specific heat capacity of silicon";
  parameter Real aTstart(unit = "K", nominal = 500.0) = 313.15;
  final parameter Real al(unit = "m") = 0.0012 "Chip length in the x direction";
  final parameter Real aw(unit = "m") = 0.0012 "Chip width in the y direction";
  final parameter Real ah(unit = "m") = 0.001 "Chip height in the z direction";
  final parameter Real aTt(unit = "K", nominal = 500.0) = 313.15 "Prescribed temperature of the top surface";
  final parameter Real aC(unit = "J/K") = 1630300.0 * al * aw * ah "Thermal capacitance of a volume";
  final parameter Real aGx(unit = "W/K") = 148.0 * aw * ah / al "Thermal conductance of a volume,x direction";
  final parameter Real aGy(unit = "W/K") = 148.0 * al * ah / aw "Thermal conductance of a volume,y direction";
  final parameter Real aGz(unit = "W/K") = 148.0 * al * aw / ah "Thermal conductance of a volume,z direction";
  Real[10, 10, 4] aT(unit = "K", start = aTstart, fixed = true, nominal = 500.0) "Temperatures of the volumes";
  Real[10, 10] aQb(unit = "W") "Power injected in the bottom volumes";
  parameter Real aPtot(unit = "W") = 100.0 "Total power consumption";
  final parameter Real aPv(unit = "W") = aPtot / 50.0 "Power dissipated in a single volume";
equation
  aQb[:,1:5] = {{aPv, aPv, aPv, aPv, aPv}, {aPv, aPv, aPv, aPv, aPv}, {aPv, aPv, aPv, aPv, aPv}, {aPv, aPv, aPv, aPv, aPv}, {aPv, aPv, aPv, aPv, aPv}, {aPv, aPv, aPv, aPv, aPv}, {aPv, aPv, aPv, aPv, aPv}, {aPv, aPv, aPv, aPv, aPv}, {aPv, aPv, aPv, aPv, aPv}, {aPv, aPv, aPv, aPv, aPv}};
  aQb[:,6:10] = {{0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}};
  der(aT[1,1,1]) = 1.0 / aC * (aGx * ((-aT[1,1,1]) + aT[2,1,1]) + aGy * ((-aT[1,1,1]) + aT[1,2,1]) + aGz * (626.3 - 3.0 * aT[1,1,1] + aT[1,1,2])) "Upper left top corner";
  der(aT[10,1,1]) = 1.0 / aC * (aGx * (aT[9,1,1] - aT[10,1,1]) + aGy * ((-aT[10,1,1]) + aT[10,2,1]) + aGz * (626.3 - 3.0 * aT[10,1,1] + aT[10,1,2])) "Lower left top corner";
  der(aT[1,10,1]) = 1.0 / aC * (aGx * ((-aT[1,10,1]) + aT[2,10,1]) + aGy * (aT[1,9,1] - aT[1,10,1]) + aGz * (626.3 - 3.0 * aT[1,10,1] + aT[1,10,2])) "Upper right top corner";
  der(aT[10,10,1]) = 1.0 / aC * (aGx * (aT[9,10,1] - aT[10,10,1]) + aGy * (aT[10,9,1] - aT[10,10,1]) + aGz * (626.3 - 3.0 * aT[10,10,1] + aT[10,10,2])) "Lower right top corner";
  der(aT[1,1,4]) = 1.0 / aC * (aGx * ((-aT[1,1,4]) + aT[2,1,4]) + aGy * ((-aT[1,1,4]) + aT[1,2,4]) + aGz * (aT[1,1,3] - aT[1,1,4]) + aQb[1,1]) "Upper left bottom corner";
  der(aT[10,1,4]) = 1.0 / aC * (aGx * (aT[9,1,4] - aT[10,1,4]) + aGy * ((-aT[10,1,4]) + aT[10,2,4]) + aGz * (aT[10,1,3] - aT[10,1,4]) + aQb[10,1]) "Lower left bottom corner";
  der(aT[1,10,4]) = 1.0 / aC * (aGx * ((-aT[1,10,4]) + aT[2,10,4]) + aGy * (aT[1,9,4] - aT[1,10,4]) + aGz * (aT[1,10,3] - aT[1,10,4]) + aQb[1,10]) "Upper right bottom corner";
  der(aT[10,10,4]) = 1.0 / aC * (aGx * (aT[9,10,1] - aT[10,10,4]) + aGy * (aT[10,9,1] - aT[10,10,4]) + aGz * (aT[10,10,3] - aT[10,10,4]) + aQb[10,10]) "Lower right bottom corner";
  for i in 2:10 - 1 loop
    der(aT[i,10,4]) = 1.0 / aC * (aGx * (aT[i - 1,10,4] - 2.0 * aT[i,10,4] + aT[i + 1,10,4]) + aGy * (aT[i,10 - 1,4] - aT[i,10,4]) + aGz * (aT[i,10,4 - 1] - aT[i,10,4]) + aQb[i,10]) "Right bottom edge";
    der(aT[i,1,4]) = 1.0 / aC * (aGx * (aT[i - 1,1,4] - 2.0 * aT[i,1,4] + aT[i + 1,1,4]) + aGy * ((-aT[i,1,4]) + aT[i,2,4]) + aGz * (aT[i,1,4 - 1] - aT[i,1,4]) + aQb[i,1]) "Left bottom edge";
    der(aT[i,10,1]) = 1.0 / aC * (aGx * (aT[i - 1,10,1] - 2.0 * aT[i,10,1] + aT[i + 1,10,1]) + aGy * (aT[i,10 - 1,1] - aT[i,10,1]) + aGz * (2.0 * 313.15 - 3.0 * aT[i,10,1] + aT[i,10,2])) "Right top edge";
    der(aT[i,1,1]) = 1.0 / aC * (aGx * (aT[i - 1,1,1] - 2.0 * aT[i,1,1] + aT[i + 1,1,1]) + aGy * ((-aT[i,1,1]) + aT[i,2,1]) + aGz * (2.0 * 313.15 - 3.0 * aT[i,1,1] + aT[i,1,2])) "Left top edge";
  end for;
  for j in 2:10 - 1 loop
    der(aT[10,j,4]) = 1.0 / aC * (aGx * (aT[10,j - 1,4] - 2.0 * aT[10,j,4] + aT[10,j + 1,4]) + aGy * (aT[10 - 1,j,4] - aT[10,j,4]) + aGz * (aT[10,j,4 - 1] - aT[10,j,4]) + aQb[10,j]) "Lower bottom edge";
    der(aT[1,j,4]) = 1.0 / aC * (aGx * (aT[1,j - 1,4] - 2.0 * aT[1,j,4] + aT[1,j + 1,4]) + aGy * ((-aT[1,j,4]) + aT[2,j,4]) + aGz * (aT[1,j,4 - 1] - aT[1,j,4]) + aQb[1,j]) "Upper bottom edge";
    der(aT[10,j,1]) = 1.0 / aC * (aGx * (aT[10,j - 1,1] - 2.0 * aT[10,j,1] + aT[10,j + 1,1]) + aGy * (aT[10 - 1,j,1] - aT[10,j,1]) + aGz * (2.0 * 313.15 - 3.0 * aT[10,j,1] + aT[10,j,2])) "Lower top edge";
    der(aT[1,j,1]) = 1.0 / aC * (aGx * (aT[1,j - 1,1] - 2.0 * aT[1,j,1] + aT[1,j + 1,1]) + aGy * ((-aT[1,j,1]) + aT[2,j,1]) + aGz * (2.0 * 313.15 - 3.0 * aT[1,j,1] + aT[1,j,2])) "Upper top edge";
  end for;
  for k in 2:4 - 1 loop
    der(aT[10,10,k]) = 1.0 / aC * (aGx * (aT[10 - 1,10,k] - aT[10,10,k]) + aGy * (aT[10,10 - 1,k] - aT[10,10,k]) + aGz * (aT[10,10,k - 1] - 2.0 * aT[10,10,k] + aT[10,10,k + 1])) "Lower right edge";
    der(aT[1,10,k]) = 1.0 / aC * (aGx * (aT[1,10 - 1,k] - aT[1,10,k]) + aGy * (aT[2,10,k] - aT[1,10,k]) + aGz * (aT[1,10,k - 1] - 2.0 * aT[1,10,k] + aT[1,10,k + 1])) "Upper right edge";
    der(aT[10,1,k]) = 1.0 / aC * (aGx * (aT[10 - 1,1,k] - aT[10,1,k]) + aGy * ((-aT[10,1,k]) + aT[10,2,k]) + aGz * (aT[10,1,k - 1] - 2.0 * aT[10,1,k] + aT[10,1,k + 1])) "Lower left edge";
    der(aT[1,1,k]) = 1.0 / aC * (aGx * ((-aT[1,1,k]) + aT[2,1,k]) + aGy * ((-aT[1,1,k]) + aT[1,2,k]) + aGz * (aT[1,1,k - 1] - 2.0 * aT[1,1,k] + aT[1,1,k + 1])) "Upper left edge";
  end for;
  for i in 2:10 - 1 loop
    for j in 2:10 - 1 loop
      der(aT[i,j,4]) = 1.0 / aC * (aGx * (aT[i - 1,j,4] - 2.0 * aT[i,j,4] + aT[i + 1,j,4]) + aGy * (aT[i,j - 1,4] - 2.0 * aT[i,j,4] + aT[i,j + 1,4]) + aGz * (aT[i,j,4 - 1] - aT[i,j,4]) + aQb[i,j]) "Bottom face";
      der(aT[i,j,1]) = 1.0 / aC * (aGx * (aT[i - 1,j,1] - 2.0 * aT[i,j,1] + aT[i + 1,j,1]) + aGy * (aT[i,j - 1,1] - 2.0 * aT[i,j,1] + aT[i,j + 1,1]) + aGz * (2.0 * 313.15 - 3.0 * aT[i,j,1] + aT[1,j,2])) "Top face";
    end for;
  end for;
  for i in 2:10 - 1 loop
    for k in 2:4 - 1 loop
      der(aT[i,10,k]) = 1.0 / aC * (aGx * (aT[i - 1,10,k] - 2.0 * aT[i,10,k] + aT[i + 1,10,k]) + aGy * (aT[i,10 - 1,k] - aT[i,10,k]) + aGz * (aT[i,10,k - 1] - 2.0 * aT[i,10,k] + aT[i,10,k + 1])) "Right face";
      der(aT[i,1,k]) = 1.0 / aC * (aGx * (aT[i - 1,1,k] - 2.0 * aT[i,1,k] + aT[i + 1,1,k]) + aGy * ((-aT[i,1,k]) + aT[i,2,k]) + aGz * (aT[i,1,k - 1] - 2.0 * aT[i,1,k] + aT[i,1,k + 1])) "Left face";
    end for;
  end for;
  for j in 2:10 - 1 loop
    for k in 2:4 - 1 loop
      der(aT[10,j,k]) = 1.0 / aC * (aGx * (aT[10 - 1,j,k] - aT[10,j,k]) + aGy * (aT[10,j - 1,k] - 2.0 * aT[10,j,k] + aT[10,j + 1,k]) + aGz * (aT[10,j,k - 1] - 2.0 * aT[10,j,k] + aT[10,j,k + 1])) "Lower face";
      der(aT[1,j,k]) = 1.0 / aC * (aGx * ((-aT[1,j,k]) + aT[2,j,k]) + aGy * (aT[1,j - 1,k] - 2.0 * aT[1,j,k] + aT[1,j + 1,k]) + aGz * (aT[1,j,k - 1] - 2.0 * aT[1,j,k] + aT[1,j,k + 1])) "Upper face";
    end for;
  end for;
  for i in 2:10 - 1 loop
    for j in 2:10 - 1 loop
      for k in 2:4 - 1 loop
        der(aT[i,j,k]) = 1.0 / aC * (aGx * (aT[i - 1,j,k] - 2.0 * aT[i,j,k] + aT[i + 1,j,k]) + aGy * (aT[i,j - 1,k] - 2.0 * aT[i,j,k] + aT[i,j + 1,k]) + aGz * (aT[i,j,k - 1] - 2.0 * aT[i,j,k] + aT[i,j,k + 1])) "Internal volume";
      end for;
    end for;
  end for;
end ThermalChipODE;
