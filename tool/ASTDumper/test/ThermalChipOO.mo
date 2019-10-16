package ThermalChipOO
  package Types
    type Temperature = Real(unit = "K", nominal = 500);
    type Power = Real(unit = "W");
    type ThermalConductivity = Real(unit = "W/(m.K)");
    type ThermalConductance = Real(unit = "W/K");
    type SpecificHeatCapacity = Real(unit = "J/(kg.K)");
    type ThermalCapacitance = Real(unit = "J/K");
    type Density = Real(unit = "kg/m3");
    type Length = Real(unit = "m");
    type Time = Real(unit = "s");
  end Types;

  package Interfaces
    connector HeatPort
      Types.Temperature T;
      flow Types.Power Q;
    end HeatPort;
  end Interfaces;

  package Models

    model Volume
      parameter Types.ThermalConductivity lambda = 148 "Thermal conductivity of silicon" annotation(
        Evaluate = true);
      parameter Types.Density rho = 2329 "Density of silicon" annotation(
        Evaluate = true);
      parameter Types.SpecificHeatCapacity c = 700 "Specific heat capacity of silicon" annotation(
        Evaluate = true);
      parameter Types.Temperature Tstart = 273.15 + 40;
      parameter Types.ThermalCapacitance C "Thermal capacitance of a volume";
      parameter Types.ThermalConductance Gx "Thermal conductance of half a volume,x direction";
      parameter Types.ThermalConductance Gy "Thermal conductance of half a volume,y direction";
      parameter Types.ThermalConductance Gz "Thermal conductance of half a volume,z 
    direction";
    
      Interfaces.HeatPort upper "Upper surface thermal port";
      Interfaces.HeatPort lower "Lower surface thermal port";
      Interfaces.HeatPort left "Left surface thermal port";
      Interfaces.HeatPort right "Right surface thermal port";
      Interfaces.HeatPort top "Top surface thermal port";
      Interfaces.HeatPort bottom "Bottom surface thermal port";
      Interfaces.HeatPort center "Volume center thermal port";
      
      Types.Temperature T "Volume temperature";
    equation
      C*der(T) = upper.Q + lower.Q + left.Q + right.Q + top.Q + bottom.Q + center.Q;
    
      upper.Q  = Gx*(upper.T  - T);
      lower.Q  = Gx*(lower.T  - T);
      left.Q   = Gy*(left.T   - T);
      right.Q  = Gy*(right.T  - T);
      top.Q    = Gz*(top.T    - T);
      bottom.Q = Gz*(bottom.T - T);
      center.T = T;
    end Volume;

    model TemperatureSource
      Interfaces.HeatPort port;
      Types.Temperature T = 298.15 "Source temperature";
    equation
  port.T = T;
    end TemperatureSource;
    
    model PowerSource
      Interfaces.HeatPort port;
      Types.Power Q = 0 "Source thermal power leaving the port";
    equation
      port.Q = -Q;
    end PowerSource;
    partial model BaseThermalChip
      parameter Integer N = 10 "Number of volumesin the x direction";
      parameter Integer M = 10 "Number of volumesin the y direction";
      parameter Integer P = 4 "Number of volumesin the z direction";
      parameter Types.Length L = 12e-3 "Chip lengthin the x direction" annotation(
        Evaluate = true);
      parameter Types.Length W = 12e-3 "Chip widthin the y direction" annotation(
        Evaluate = true);
      parameter Types.Length H = 4e-3 "Chip heightin the z direction" annotation(
        Evaluate = true);
      parameter Types.ThermalConductivity lambda = 148 "Thermal conductivity of silicon" annotation(
        Evaluate = true);
      parameter Types.Density rho = 2329 "Density of silicon" annotation(
        Evaluate = true);
      parameter Types.SpecificHeatCapacity c = 700 "Specific heat capacity of silicon" annotation(
        Evaluate = true);
      parameter Types.Temperature Tstart = 273.15 + 40;
      final parameter Types.Length l = L / N "Chip lengthin the x direction";
      final parameter Types.Length w = W / M "Chip widthin the y direction";
      final parameter Types.Length h = H / P "Chip heightin the z direction";
      parameter Types.Temperature Tt = 273.15 + 40 "Prescribed temperature of the top surface" annotation(
        Evaluate = true);
      final parameter Types.ThermalCapacitance C = rho*c*l*w*h "Thermal capacitance of a volume";
      final parameter Types.ThermalConductance Gx = lambda*w*h/l "Thermal conductance of a volume,x direction";
      final parameter Types.ThermalConductance Gy = lambda*l*h/w "Thermal conductance of a volume,y direction";
      final parameter Types.ThermalConductance Gz = lambda*l*w/h "Thermal conductance of a volume,z direction";
    
      Volume vol[N,M,P](each T(start = Tstart, fixed = true),
                        each C = C,
                        each Gx = 2*Gx, each Gy = 2*Gy, each Gz = 2*Gz);
      TemperatureSource Tsource[N,M](each T = Tt);
    equation
      // Connections in the z direction
      for i in 1:N loop
        for j in 1:M loop
          connect(vol[i,j,1].top, Tsource[i,j].port);
          for k in 1:P-1 loop
            connect(vol[i,j,k].bottom, vol[i,j,k+1].top);
          end for;
        end for;
      end for;
      
      // Connections in the y direction
      for i in 1:N loop
        for k in 1:P loop
          for j in 1:M-1 loop
            connect(vol[i,j,k].right, vol[i,j+1,k].left);
          end for;
        end for;
      end for;
    
      // Connections in the x direction
      for j in 1:M loop
        for k in 1:P loop
           for i in 1:N-1 loop
             connect(vol[i,j,k].lower, vol[i+1,j,k].upper);
           end for;
         end for;
       end for;
    end BaseThermalChip;

    model ThermalChip4Cores "Thermal chip model written as ODE with 4 simulated cores"
      extends BaseThermalChip(
        final N = 12*Nr, final M = 12*Nr, final P = 4*Pr);
      parameter Types.Time Ts = 10e-3 "Switching base period";
      parameter Types.Power Ptot = 100 "Total average power consumption";
      final parameter Types.Power Pc = Ptot/4 "Average power dissipated by each core";
      final parameter Types.Power Pa = Pc/4 "Average power dissipated by each area in a core";
      final parameter Types.Power Pv=Pa/(4*Nr^2) "Average power dissipated in a single volume";
      final parameter Types.Power Pvmax = Pv/(sum(AS)/32*sum(CS)/32) "Max power dissipated in a single volume";
      parameter Integer Nr=1 "Grid refining factor on x-y plane";
      parameter Integer Pr=1 "Grid refining factor on z direction";
      parameter Integer TLC[4,2] = 
        {{1,1},
         {1,7},
         {7,1},
         {7,7}
        } "Upper-left coordinate of each core on base grid";
      parameter Integer TLA[4,2] =
        {{0,0},
         {2,0},
         {0,2},
         {2,2}
        }
      "Upper left coordinate of each area on base grid, relative to core"; 
      parameter Integer AS[32] = 
        {1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1}
        "Single area switching sequence"
        annotation(Evaluate=true);
      parameter Integer CS[32] = 
        {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0}
        "Core switching sequence"
        annotation(Evaluate = true);
      Integer a_idx[4](start = {1, 9, 17, 25}, each fixed = true) "Sequence index of each area";
      Integer c_idx[4](start = {1, 9, 17, 25}, each fixed = true) "Sequence index of each core";
      Integer cs_ctr(start = 1, fixed = true) "Core switching counter";
      Types.Power Qv[4,4] "Power dissipated in each area of each core, per single volume";
      PowerSource Qsource[4,4,2,2] "Array of power sources, first two indeces are core and area";
    algorithm
    //Switching sequence
      when sample(0, Ts) then
        for i in 1:4 loop    // core loop
          for j in 1:4 loop  // area loop
            a_idx[j] := mod((a_idx[j]),32) + 1;
            Qv[i,j] := Pvmax*AS[a_idx[j]]*CS[c_idx[i]];
          end for;
        end for;
        if mod(pre(cs_ctr),32) == 0 then
          for i in 1:4 loop
            c_idx[i] := mod((c_idx[i]),32) + 1;
          end for;
        end if;
        cs_ctr := cs_ctr+1;
      end when;
    equation
      for i in 1:4 loop  // core loop
        for j in 1:4 loop     // area loop
          Qsource[i,j,:,:].Q = fill(Qv[i,j], 2*Nr, 2*Nr);
          connect(Qsource[i,j,:,:].port, 
                  vol[(TLC[i,1]+TLA[j,1])*Nr+1:(TLC[i,1]+TLA[j,1]+2)*Nr,
                      (TLC[i,2]+TLA[j,2])*Nr+1:(TLC[i,2]+TLA[j,2]+2)*Nr, P].center);
        end for;
      end for;
    annotation(
        experiment(StartTime = 0, StopTime = 4, Tolerance = 1e-6, Interval = 0.001),
        __OpenModelica_simulationFlags(lv = "LOG_STATS", outputFormat = "mat", s = "euler"));
    end ThermalChip4Cores;

    model ThermalChipSimpleBoundary "Thermal chip model written by explicit ODEs, constant power on half of the bottom surface"
      extends BaseThermalChip;
      parameter Types.Power Ptot = 100 "Total power consumption";
      final parameter Types.Power Pv = Ptot / (N * M / 2) "Power dissipated in a single volume";
      PowerSource Qsource[N,div(M,2)](each Q = Pv);
    equation
      connect(Qsource.port, vol[:, 1:div(M,2), P].center);
      annotation(
        experiment(StartTime = 0, StopTime = 0.01, Tolerance = 1e-6, Interval = 0.000002),
        __OpenModelica_simulationFlags(lv = "LOG_STATS", outputFormat = "mat", s = "euler"));
    end ThermalChipSimpleBoundary;
  end Models;

  package Benchmarks
  end Benchmarks;
end ThermalChipOO;
