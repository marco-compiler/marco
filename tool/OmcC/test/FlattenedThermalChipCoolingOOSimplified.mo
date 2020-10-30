class ThermalChipOO
  final parameter Integer N = 4 "Number of volumesin the x direction";
  final parameter Integer M = 4 "Number of volumesin the y direction";
  final parameter Integer P = 4 "Number of volumesin the z direction";
  final parameter Real L(unit = "m") = 0.012 "Chip lengthin the x direction";
  final parameter Real W(unit = "m") = 0.012 "Chip widthin the y direction";
  final parameter Real H(unit = "m") = 0.004 "Chip heightin the z direction";
  final parameter Real lambda(unit = "W/(mK)") = 148.0 "Thermal conductivity of silicon";
  final parameter Real rho(unit = "kg/m3") = 2329.0 "Density of silicon";
  final parameter Real c(unit = "J/(kgK)") = 700.0 "Specific heat capacity of silicon";
  parameter Real Tstart(unit = "K", nominal = 500.0) = 313.15;
  final parameter Real l(unit = "m") = 0.003 "Chip lengthin the x direction";
  final parameter Real w(unit = "m") = 0.003 "Chip widthin the y direction";
  final parameter Real h(unit = "m") = 0.001 "Chip heightin the z direction";
  final parameter Real Tt(unit = "K", nominal = 500.0) = 313.15 "Prescribed temperature of the top surface";
  final parameter Real C(unit = "J/K") = 1630300.0 * l * w * h "Thermal capacitance of a volume";
  final parameter Real Gx(unit = "W/K") = 148.0 * w * h / l "Thermal conductance of a volume,x direction";
  final parameter Real Gy(unit = "W/K") = 148.0 * l * h / w "Thermal conductance of a volume,y direction";
  final parameter Real Gz(unit = "W/K") = 148.0 * l * w / h "Thermal conductance of a volume,z direction";
  Real[4, 4, 4] volT(unit = "K", start = Tstart, fixed = true, nominal = 500.0) "Volume temperature";
  Real[4, 4, 4] volcenterQ(unit = "W");
  Real[4, 4, 4] volcenterT(unit = "K", nominal = 500.0);
  Real[4, 4, 4] volbottomQ(unit = "W");
  Real[4, 4, 4] volbottomT(unit = "K", nominal = 500.0);
  Real[4, 4, 4] voltopQ(unit = "W");
  Real[4, 4, 4] voltopT(unit = "K", nominal = 500.0);
  Real[4, 4, 4] volrightQ(unit = "W");
  Real[4, 4, 4] volrightT(unit = "K", nominal = 500.0);
  Real[4, 4, 4] volleftQ(unit = "W");
  Real[4, 4, 4] volleftT(unit = "K", nominal = 500.0);
  Real[4, 4, 4] vollowerQ(unit = "W");
  Real[4, 4, 4] vollowerT(unit = "K", nominal = 500.0);
  Real[4, 4, 4] volupperQ(unit = "W");
  Real[4, 4, 4] volupperT(unit = "K", nominal = 500.0);
  parameter Real[4, 4, 4] volGz(unit = "W/K", start=2.0 * Gz) "Thermal conductance of half a volume,z 
     direction";
  parameter Real[4, 4, 4] volGy(unit = "W/K", start=2.0 * Gy) "Thermal conductance of half a volume,y direction";
  parameter Real[4, 4, 4] volGx(unit = "W/K", start= 2.0 * Gx) "Thermal conductance of half a volume,x direction";
  parameter Real[4, 4, 4] volC(unit = "J/K", start= C) "Thermal capacitance of a volume";
  parameter Real[4, 4, 4] volTstart(unit = "K", nominal = 500.0, start=313.15);
  final parameter Real[4, 4, 4] volc(unit = "J/(kgK)", start = 700.0) "Specific heat capacity of silicon";
  final parameter Real[4, 4, 4] volrho(unit = "kg/m3", start= 2329.0) "Density of silicon";
  final parameter Real[4, 4, 4] vollambda(unit = "W/(mK)", start= 148.0) "Thermal conductivity of silicon";
  parameter Real[4, 4] TsourceT(unit = "K", nominal = 500.0, start=313.15) "Source temperature";
  Real[4, 4] TsourceportQ(unit = "W");
  Real[4, 4] TsourceportT(unit = "K", nominal = 500.0);
  parameter Real Ptot(unit = "W") = 100.0 "Total power consumption";
  final parameter Real Pv(unit = "W") = Ptot / 8.0 "Power dissipated in a single volume";
  parameter Real[4, 2] QsourceQ(unit = "W", start=Pv) "Source thermal power leaving the port";
  Real[4, 2] QsourceportQ(unit = "W");
  Real[4, 2] QsourceportT(unit = "K", nominal = 500.0);
equation
  for i1 in 1:4 loop
    for i2 in 1:4 loop
      for i3 in 1:4 loop
        volC[i1,i2,i3] * der(volT[i1,i2,i3]) = volupperQ[i1,i2,i3] + vollowerQ[i1,i2,i3] + volleftQ[i1,i2,i3] + volrightQ[i1,i2,i3] + voltopQ[i1,i2,i3] + volbottomQ[i1,i2,i3] + volcenterQ[i1,i2,i3];
      end for;
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 1:4 loop
      for i3 in 1:4 loop
        volupperQ[i1,i2,i3] = volGx[i1,i2,i3] * (volupperT[i1,i2,i3] - volT[i1,i2,i3]);
      end for;
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 1:4 loop
      for i3 in 1:4 loop
        vollowerQ[i1,i2,i3] = volGx[i1,i2,i3] * (vollowerT[i1,i2,i3] - volT[i1,i2,i3]);
      end for;
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 1:4 loop
      for i3 in 1:4 loop
        volleftQ[i1,i2,i3] = volGy[i1,i2,i3] * (volleftT[i1,i2,i3] - volT[i1,i2,i3]);
      end for;
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 1:4 loop
      for i3 in 1:4 loop
        volrightQ[i1,i2,i3] = volGy[i1,i2,i3] * (volrightT[i1,i2,i3] - volT[i1,i2,i3]);
      end for;
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 1:4 loop
      for i3 in 1:4 loop
        voltopQ[i1,i2,i3] = volGz[i1,i2,i3] * (voltopT[i1,i2,i3] - volT[i1,i2,i3]);
      end for;
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 1:4 loop
      for i3 in 1:4 loop
        volbottomQ[i1,i2,i3] = volGz[i1,i2,i3] * (volbottomT[i1,i2,i3] - volT[i1,i2,i3]);
      end for;
    end for;
  end for;
  volcenterT = volT;
  TsourceportT = TsourceT;
  for i1 in 1:4 loop
    for i2 in 1:2 loop
      QsourceportQ[i1,i2] = -QsourceQ[i1,i2];
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 1:4 loop
      TsourceportT[i1,i2] = voltopT[i1,i2,1];
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 1:4 loop
      voltopQ[i1,i2,1] + TsourceportQ[i1,i2] = 0.0;
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 1:4 loop
      for i3 in 2:4 loop
        voltopT[i1,i2,i3] = volbottomT[i1,i2,i3 - 1];
      end for;
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 1:4 loop
      for i3 in 1:3 loop
        volbottomQ[i1,i2,i3] + voltopQ[i1,i2,i3 + 1] = 0.0;
      end for;
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 2:4 loop
      for i3 in 1:4 loop
        volleftT[i1,i2,i3] = volrightT[i1,i2 - 1,i3];
      end for;
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 1:3 loop
      for i3 in 1:4 loop
        volrightQ[i1,i2,i3] + volleftQ[i1,i2 + 1,i3] = 0.0;
      end for;
    end for;
  end for;
  for i1 in 2:4 loop
    for i2 in 1:4 loop
      for i3 in 1:4 loop
        volupperT[i1,i2,i3] = vollowerT[i1 - 1,i2,i3];
      end for;
    end for;
  end for;
  for i1 in 1:3 loop
    for i2 in 1:4 loop
      for i3 in 1:4 loop
        vollowerQ[i1,i2,i3] + volupperQ[i1 + 1,i2,i3] = 0.0;
      end for;
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 1:2 loop
      QsourceportT[i1,i2] = volcenterT[i1,i2,4];
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 1:2 loop
      QsourceportQ[i1,i2] + volcenterQ[i1,i2,4] = 0.0;
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 3:4 loop
      for i3 in 1:4 loop
        volcenterQ[i1,i2,i3] = 0.0;
      end for;
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 1:2 loop
      for i3 in 1:3 loop
        volcenterQ[i1,i2,i3] = 0.0;
      end for;
    end for;
  end for;
  for i1 in 1:4 loop
    for i2 in 1:4 loop
      volbottomQ[i1,i2,4] = 0.0;
    end for;
  end for;
  for i1 in 1:4 loop
    for i3 in 1:4 loop
      volrightQ[i1,4,i3] = 0.0;
    end for;
  end for;
  for i1 in 1:4 loop
    for i3 in 1:4 loop
      volleftQ[i1,1,i3] = 0.0;
    end for;
  end for;
  for i2 in 1:4 loop
    for i3 in 1:4 loop
      vollowerQ[4,i2,i3] = 0.0;
    end for;
  end for;
  for i2 in 1:4 loop
    for i3 in 1:4 loop
      volupperQ[1,i2,i3] = 0.0;
    end for;
  end for;
end ThermalChipOO;
