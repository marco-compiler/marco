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
  final parameter Real ZaTt(unit = "K", nominal = 500.0) = 313.15 "Prescribed temperature of the top surface";
  final parameter Real Zac(unit = "J/K") = 1630300.0 * al * aw * ah "Thermal capacitance of a volume";
  final parameter Real ZaGx(unit = "W/K") = 148.0 * aw * ah / al "Thermal conductance of a volume,x direction";
  final parameter Real ZaGy(unit = "W/K") = 148.0 * al * ah / aw "Thermal conductance of a volume,y direction";
  final parameter Real ZaGz(unit = "W/K") = 148.0 * al * aw / ah "Thermal conductance of a volume,z direction";
  Real[10, 10, 4] ZaT(unit = "K", start = aTstart, fixed = true, nominal = 500.0) "Temperatures of the volumes";
  Real[10, 10] ZaQb(unit = "W") "Power injected in the bottom volumes";
  parameter Real aPtot(unit = "W") = 100.0 "Total power consumption";
  final parameter Real ZaPv(unit = "W") = aPtot / 50.0 "Power dissipated in a single volume";
equation
  for i in 1:10 loop
    for k in 1:5 loop
	  ZaQb[i,k] = ZaPv;
	end for;
end for;
for i in 1:10 loop
	for k in 6:10 loop
	  ZaQb[i,k] = 0.0;
	end for;
end for;
  der(ZaT[1,1,1]) = 1.0 / Zac * (ZaGx * ((-ZaT[1,1,1]) + ZaT[2,1,1]) + ZaGy * ((-ZaT[1,1,1]) + ZaT[1,2,1]) + ZaGz * (626.3 - 3.0 * ZaT[1,1,1] + ZaT[1,1,2])) "Upper left top corner";
  der(ZaT[10,1,1]) = 1.0 / Zac * (ZaGx * (ZaT[9,1,1] - ZaT[10,1,1]) + ZaGy * ((-ZaT[10,1,1]) + ZaT[10,2,1]) + ZaGz * (626.3 - 3.0 * ZaT[10,1,1] + ZaT[10,1,2])) "Lower left top corner";
  der(ZaT[1,10,1]) = 1.0 / Zac * (ZaGx * ((-ZaT[1,10,1]) + ZaT[2,10,1]) + ZaGy * (ZaT[1,9,1] - ZaT[1,10,1]) + ZaGz * (626.3 - 3.0 * ZaT[1,10,1] + ZaT[1,10,2])) "Upper right top corner";
  der(ZaT[10,10,1]) = 1.0 / Zac * (ZaGx * (ZaT[9,10,1] - ZaT[10,10,1]) + ZaGy * (ZaT[10,9,1] - ZaT[10,10,1]) + ZaGz * (626.3 - 3.0 * ZaT[10,10,1] + ZaT[10,10,2])) "Lower right top corner";
  der(ZaT[1,1,4]) = 1.0 / Zac * (ZaGx * ((-ZaT[1,1,4]) + ZaT[2,1,4]) + ZaGy * ((-ZaT[1,1,4]) + ZaT[1,2,4]) + ZaGz * (ZaT[1,1,3] - ZaT[1,1,4]) + ZaQb[1,1]) "Upper left bottom corner";
  der(ZaT[10,1,4]) = 1.0 / Zac * (ZaGx * (ZaT[9,1,4] - ZaT[10,1,4]) + ZaGy * ((-ZaT[10,1,4]) + ZaT[10,2,4]) + ZaGz * (ZaT[10,1,3] - ZaT[10,1,4]) + ZaQb[10,1]) "Lower left bottom corner";
  der(ZaT[1,10,4]) = 1.0 / Zac * (ZaGx * ((-ZaT[1,10,4]) + ZaT[2,10,4]) + ZaGy * (ZaT[1,9,4] - ZaT[1,10,4]) + ZaGz * (ZaT[1,10,3] - ZaT[1,10,4]) + ZaQb[1,10]) "Upper right bottom corner";
  der(ZaT[10,10,4]) = 1.0 / Zac * (ZaGx * (ZaT[9,10,1] - ZaT[10,10,4]) + ZaGy * (ZaT[10,9,1] - ZaT[10,10,4]) + ZaGz * (ZaT[10,10,3] - ZaT[10,10,4]) + ZaQb[10,10]) "Lower right bottom corner";
  for i in 2:9 loop
    der(ZaT[i,10,4]) = 1.0 / Zac * (ZaGx * (ZaT[i - 1,10,4] - 2.0 * ZaT[i,10,4] + ZaT[i + 1,10,4]) + ZaGy * (ZaT[i,10 - 1,4] - ZaT[i,10,4]) + ZaGz * (ZaT[i,10,4 - 1] - ZaT[i,10,4]) + ZaQb[i,10]) "Right bottom edge";
    der(ZaT[i,1,4]) = 1.0 / Zac * (ZaGx * (ZaT[i - 1,1,4] - 2.0 * ZaT[i,1,4] + ZaT[i + 1,1,4]) + ZaGy * ((-ZaT[i,1,4]) + ZaT[i,2,4]) + ZaGz * (ZaT[i,1,4 - 1] - ZaT[i,1,4]) + ZaQb[i,1]) "Left bottom edge";
    der(ZaT[i,10,1]) = 1.0 / Zac * (ZaGx * (ZaT[i - 1,10,1] - 2.0 * ZaT[i,10,1] + ZaT[i + 1,10,1]) + ZaGy * (ZaT[i,10 - 1,1] - ZaT[i,10,1]) + ZaGz * (2.0 * 313.15 - 3.0 * ZaT[i,10,1] + ZaT[i,10,2])) "Right top edge";
    der(ZaT[i,1,1]) = 1.0 / Zac * (ZaGx * (ZaT[i - 1,1,1] - 2.0 * ZaT[i,1,1] + ZaT[i + 1,1,1]) + ZaGy * ((-ZaT[i,1,1]) + ZaT[i,2,1]) + ZaGz * (2.0 * 313.15 - 3.0 * ZaT[i,1,1] + ZaT[i,1,2])) "Left top edge";
  end for;
  for j in 2:9 loop
    der(ZaT[10,j,4]) = 1.0 / Zac * (ZaGx * (ZaT[10,j - 1,4] - 2.0 * ZaT[10,j,4] + ZaT[10,j + 1,4]) + ZaGy * (ZaT[10 - 1,j,4] - ZaT[10,j,4]) + ZaGz * (ZaT[10,j,4 - 1] - ZaT[10,j,4]) + ZaQb[10,j]) "Lower bottom edge";
    der(ZaT[1,j,4]) = 1.0 / Zac * (ZaGx * (ZaT[1,j - 1,4] - 2.0 * ZaT[1,j,4] + ZaT[1,j + 1,4]) + ZaGy * ((-ZaT[1,j,4]) + ZaT[2,j,4]) + ZaGz * (ZaT[1,j,4 - 1] - ZaT[1,j,4]) + ZaQb[1,j]) "Upper bottom edge";
    der(ZaT[10,j,1]) = 1.0 / Zac * (ZaGx * (ZaT[10,j - 1,1] - 2.0 * ZaT[10,j,1] + ZaT[10,j + 1,1]) + ZaGy * (ZaT[10 - 1,j,1] - ZaT[10,j,1]) + ZaGz * (2.0 * 313.15 - 3.0 * ZaT[10,j,1] + ZaT[10,j,2])) "Lower top edge";
    der(ZaT[1,j,1]) = 1.0 / Zac * (ZaGx * (ZaT[1,j - 1,1] - 2.0 * ZaT[1,j,1] + ZaT[1,j + 1,1]) + ZaGy * ((-ZaT[1,j,1]) + ZaT[2,j,1]) + ZaGz * (2.0 * 313.15 - 3.0 * ZaT[1,j,1] + ZaT[1,j,2])) "Upper top edge";
  end for;
  for k in 2:3 loop
    der(ZaT[10,10,k]) = 1.0 / Zac * (ZaGx * (ZaT[10 - 1,10,k] - ZaT[10,10,k]) + ZaGy * (ZaT[10,10 - 1,k] - ZaT[10,10,k]) + ZaGz * (ZaT[10,10,k - 1] - 2.0 * ZaT[10,10,k] + ZaT[10,10,k + 1])) "Lower right edge";
    der(ZaT[1,10,k]) = 1.0 / Zac * (ZaGx * (ZaT[1,10 - 1,k] - ZaT[1,10,k]) + ZaGy * (ZaT[2,10,k] - ZaT[1,10,k]) + ZaGz * (ZaT[1,10,k - 1] - 2.0 * ZaT[1,10,k] + ZaT[1,10,k + 1])) "Upper right edge";
    der(ZaT[10,1,k]) = 1.0 / Zac * (ZaGx * (ZaT[10 - 1,1,k] - ZaT[10,1,k]) + ZaGy * ((-ZaT[10,1,k]) + ZaT[10,2,k]) + ZaGz * (ZaT[10,1,k - 1] - 2.0 * ZaT[10,1,k] + ZaT[10,1,k + 1])) "Lower left edge";
    der(ZaT[1,1,k]) = 1.0 / Zac * (ZaGx * ((-ZaT[1,1,k]) + ZaT[2,1,k]) + ZaGy * ((-ZaT[1,1,k]) + ZaT[1,2,k]) + ZaGz * (ZaT[1,1,k - 1] - 2.0 * ZaT[1,1,k] + ZaT[1,1,k + 1])) "Upper left edge";
  end for;
  for i in 2:9 loop
    for j in 2:9 loop
      der(ZaT[i,j,4]) = 1.0 / Zac * (ZaGx * (ZaT[i - 1,j,4] - 2.0 * ZaT[i,j,4] + ZaT[i + 1,j,4]) + ZaGy * (ZaT[i,j - 1,4] - 2.0 * ZaT[i,j,4] + ZaT[i,j + 1,4]) + ZaGz * (ZaT[i,j,4 - 1] - ZaT[i,j,4]) + ZaQb[i,j]) "Bottom face";
      der(ZaT[i,j,1]) = 1.0 / Zac * (ZaGx * (ZaT[i - 1,j,1] - 2.0 * ZaT[i,j,1] + ZaT[i + 1,j,1]) + ZaGy * (ZaT[i,j - 1,1] - 2.0 * ZaT[i,j,1] + ZaT[i,j + 1,1]) + ZaGz * (2.0 * 313.15 - 3.0 * ZaT[i,j,1] + ZaT[1,j,2])) "Top face";
    end for;
  end for;
  for i in 2:9 loop
    for k in 2:3 loop
      der(ZaT[i,10,k]) = 1.0 / Zac * (ZaGx * (ZaT[i - 1,10,k] - 2.0 * ZaT[i,10,k] + ZaT[i + 1,10,k]) + ZaGy * (ZaT[i,10 - 1,k] - ZaT[i,10,k]) + ZaGz * (ZaT[i,10,k - 1] - 2.0 * ZaT[i,10,k] + ZaT[i,10,k + 1])) "Right face";
      der(ZaT[i,1,k]) = 1.0 / Zac * (ZaGx * (ZaT[i - 1,1,k] - 2.0 * ZaT[i,1,k] + ZaT[i + 1,1,k]) + ZaGy * ((-ZaT[i,1,k]) + ZaT[i,2,k]) + ZaGz * (ZaT[i,1,k - 1] - 2.0 * ZaT[i,1,k] + ZaT[i,1,k + 1])) "Left face";
    end for;
  end for;
  for j in 2:9 loop
    for k in 2:3 loop
      der(ZaT[10,j,k]) = 1.0 / Zac * (ZaGx * (ZaT[10 - 1,j,k] - ZaT[10,j,k]) + ZaGy * (ZaT[10,j - 1,k] - 2.0 * ZaT[10,j,k] + ZaT[10,j + 1,k]) + ZaGz * (ZaT[10,j,k - 1] - 2.0 * ZaT[10,j,k] + ZaT[10,j,k + 1])) "Lower face";
      der(ZaT[1,j,k]) = 1.0 / Zac * (ZaGx * ((-ZaT[1,j,k]) + ZaT[2,j,k]) + ZaGy * (ZaT[1,j - 1,k] - 2.0 * ZaT[1,j,k] + ZaT[1,j + 1,k]) + ZaGz * (ZaT[1,j,k - 1] - 2.0 * ZaT[1,j,k] + ZaT[1,j,k + 1])) "Upper face";
    end for;
  end for;
  for i in 2:9 loop
    for j in 2:9 loop
      for k in 2:3 loop
        der(ZaT[i,j,k]) = 1.0 / Zac * (ZaGx * (ZaT[i - 1,j,k] - 2.0 * ZaT[i,j,k] + ZaT[i + 1,j,k]) + ZaGy * (ZaT[i,j - 1,k] - 2.0 * ZaT[i,j,k] + ZaT[i,j + 1,k]) + ZaGz * (ZaT[i,j,k - 1] - 2.0 * ZaT[i,j,k] + ZaT[i,j,k + 1])) "Internal volume";
      end for;
    end for;
  end for;
end ThermalChipODE;
