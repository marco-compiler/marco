package ThermalChipODE
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
  end Interfaces;

  package Models
    partial model BaseThermalChip
      parameter Integer N = 10 "Number of volumes in the x direction";
      parameter Integer M = 10 "Number of volumes in the y direction";
      parameter Integer P = 4 "Number of volumes in the z direction";
      parameter Types.Length L = 12e-3 "Chip length in the x direction" annotation(
        Evaluate = true);
      parameter Types.Length W = 12e-3 "Chip width in the y direction" annotation(
        Evaluate = true);
      parameter Types.Length H = 4e-3 "Chip height in the z direction" annotation(
        Evaluate = true);
      parameter Types.ThermalConductivity lambda = 148 "Thermal conductivity of silicon" annotation(
        Evaluate = true);
      parameter Types.Density rho = 2329 "Density of silicon" annotation(
        Evaluate = true);
      parameter Types.SpecificHeatCapacity c = 700 "Specific heat capacity of silicon" annotation(
        Evaluate = true);
      parameter Types.Temperature Tstart = 273.15 + 40;
      final parameter Types.Length l = L / N "Chip length in the x direction";
      final parameter Types.Length w = W / M "Chip width in the y direction";
      final parameter Types.Length h = H / P "Chip height in the z direction";
      parameter Types.Temperature Tt = 273.15 + 40 "Prescribed temperature of the top surface" annotation(
        Evaluate = true);
      final parameter Types.ThermalCapacitance C = rho*c*l*w*h "Thermal capacitance of a volume";
      final parameter Types.ThermalConductance Gx = lambda*w*h / l "Thermal conductance of a volume,x direction";
      final parameter Types.ThermalConductance Gy = lambda*l*h / w "Thermal conductance of a volume,y direction";
      final parameter Types.ThermalConductance Gz = lambda*l*w / h "Thermal conductance of a volume,z direction";
      Types.Temperature T[N,M,P](each start = Tstart,each fixed = true) "Temperatures of the volumes";
      Types.Power Qb[N,M] "Power injected in the bottom volumes";
    equation
      der(T[1,1,1]) = 1/C*(Gx*((-T[1,1,1]) + T[2,1,1]) +
                           Gy*((-T[1,1,1]) + T[1,2,1]) +
                           Gz*(2*Tt-3*T[1,1,1] + T[1,1,2])) "Upper left top corner";
    
      der(T[N,1,1]) = 1/C*(Gx*(T[N-1,1,1]-T[N,1,1]) +
                           Gy*((-T[N,1,1]) + T[N,2,1]) +
                           Gz*(2*Tt-3*T[N,1,1] + T[N,1,2])) "Lower left top corner";
    
      der(T[1,M,1]) = 1/C*(Gx*((-T[1,M,1]) + T[2,M,1]) +
                           Gy*(T[1,M-1,1]-T[1,M,1]) +
                           Gz*(2*Tt-3*T[1,M,1] + T[1,M,2])) "Upper right top corner";
    
      der(T[N,M,1]) = 1/C*(Gx*(T[N-1,M,1]-T[N,M,1]) +
                           Gy*(T[N,M-1,1]-T[N,M,1]) +
                           Gz*(2*Tt-3*T[N,M,1] + T[N,M,2])) "Lower right top corner";
    
      der(T[1,1,P]) = 1/C*(Gx*((-T[1,1,P]) + T[2,1,P]) +
                           Gy*((-T[1,1,P]) + T[1,2,P]) +
                           Gz*(T[1,1,P-1]-T[1,1,P]) + Qb[1,1]) "Upper left bottom corner";
    
      der(T[N,1,P]) = 1/C*(Gx*(T[N-1,1,P]-T[N,1,P]) +
                           Gy*((-T[N,1,P]) + T[N,2,P]) +
                           Gz*(T[N,1,P-1]-T[N,1,P]) + Qb[N,1]) "Lower left bottom corner";
    
      der(T[1,M,P]) = 1/C*(Gx*((-T[1,M,P]) + T[2,M,P]) +
                           Gy*(T[1,M-1,P]-T[1,M,P]) +
                           Gz*(T[1,M,P-1]-T[1,M,P]) + Qb[1,M]) "Upper right bottom corner";
    
      der(T[N,M,P]) = 1/C*(Gx*(T[N-1,M,1]-T[N,M,P]) +
                           Gy*(T[N,M-1,1]-T[N,M,P]) + 
                           Gz*(T[N,M,P-1]-T[N,M,P]) + Qb[N,M]) "Lower right bottom corner";
    
      for i in 2:N-1 loop
        der(T[i,1,1]) = 1/C*(Gx*(T[i-1,1,1]-2*T[i,1,1] + T[i+1,1,1]) +
                             Gy*((-T[i,1,1]) + T[i,2,1]) +
                             Gz*(2*Tt-3*T[i,1,1] + T[i,1,2])) "Left top edge";
    
        der(T[i,M,1]) = 1/C*(Gx*(T[i-1,M,1]-2*T[i,M,1] + T[i+1,M,1]) +
                             Gy*(T[i,M-1,1]-T[i,M,1]) +
                             Gz*(2*Tt-3*T[i,M,1] + T[i,M,2])) "Right top edge";
    
        der(T[i,1,P]) = 1/C*(Gx*(T[i-1,1,P]-2*T[i,1,P] + T[i+1,1,P]) +
                             Gy*((-T[i,1,P]) + T[i,2,P]) +
                             Gz*(T[i,1,P-1]-T[i,1,P]) + Qb[i,1]) "Left bottom edge";
    
        der(T[i,M,P]) = 1/C*(Gx*(T[i-1,M,P]-2*T[i,M,P] + T[i+1,M,P]) +
                             Gy*(T[i,M-1,P]-T[i,M,P]) +
                             Gz*(T[i,M,P-1]-T[i,M,P]) + Qb[i,M]) "Right bottom edge";
      end for;
    
      for j in 2:M-1 loop
        der(T[1,j,1]) = 1/C*(Gx*(T[1,j-1,1]-2*T[1,j,1] + T[1,j+1,1]) +
                             Gy*((-T[1,j,1]) + T[2,j,1]) +
                             Gz*(2*Tt-3*T[1,j,1] + T[1,j,2])) "Upper top edge";
    
        der(T[N,j,1]) = 1/C*(Gx*(T[N,j-1,1]-2*T[N,j,1] + T[N,j+1,1]) +
                             Gy*(T[N-1,j,1]-T[N,j,1]) +
                             Gz*(2*Tt-3*T[N,j,1] + T[N,j,2])) "Lower top edge";
    
        der(T[1,j,P]) = 1/C*(Gx*(T[1,j-1,P]-2*T[1,j,P] + T[1,j+1,P]) +
                             Gy*((-T[1,j,P]) + T[2,j,P]) +
                             Gz*(T[1,j,P-1]-T[1,j,P]) + Qb[1,j]) "Upper bottom edge";
    
        der(T[N,j,P]) = 1/C*(Gx*(T[N,j-1,P]-2*T[N,j,P] + T[N,j+1,P]) +
                             Gy*(T[N-1,j,P]-T[N,j,P]) +
                             Gz*(T[N,j,P-1]-T[N,j,P]) + Qb[N,j]) "Lower bottom edge";
      end for;
    
      for k in 2:P-1 loop
        der(T[1,1,k]) = 1/C*(Gx*((-T[1,1,k]) + T[2,1,k]) +
                             Gy*((-T[1,1,k]) + T[1,2,k]) +
                             Gz*(T[1,1,k-1]-2*T[1,1,k] + T[1,1,k + 1])) "Upper left edge";
    
        der(T[N,1,k]) = 1/C*(Gx*(T[N-1,1,k]-T[N,1,k]) +
                             Gy*((-T[N,1,k]) + T[N,2,k]) +
                             Gz*(T[N,1,k-1]-2*T[N,1,k] + T[N,1,k + 1])) "Lower left edge";
    
        der(T[1,M,k]) = 1/C*(Gx*(T[1,M-1,k]-T[1,M,k]) +
                             Gy*(T[2,M,k]-T[1,M,k]) +
                             Gz*(T[1,M,k-1]-2*T[1,M,k] + T[1,M,k + 1])) "Upper right edge";
    
        der(T[N,M,k]) = 1/C*(Gx*(T[N-1,M,k]-T[N,M,k]) +
                             Gy*(T[N,M-1,k]-T[N,M,k]) +
                             Gz*(T[N,M,k-1]-2*T[N,M,k] + T[N,M,k + 1])) "Lower right edge";
      end for;
    
      for i in 2:N-1 loop
        for j in 2:M-1 loop
          der(T[i,j,1]) = 1/C*(Gx*(T[i-1,j,1]-2*T[i,j,1] + T[i+1,j,1]) +
                               Gy*(T[i,j-1,1]-2*T[i,j,1] + T[i,j+1,1]) +
                               Gz*(2*Tt-3*T[i,j,1] + T[1,j,2])) "Top face";
    
          der(T[i,j,P]) = 1/C*(Gx*(T[i-1,j,P]-2*T[i,j,P] + T[i+1,j,P]) +
                               Gy*(T[i,j-1,P]-2*T[i,j,P] + T[i,j+1,P]) +
                               Gz*(T[i,j,P-1]-T[i,j,P]) + Qb[i,j]) "Bottom face";
        end for;
      end for;
    
      for i in 2:N-1 loop
        for k in 2:P-1 loop
          der(T[i,1,k]) = 1/C*(Gx*(T[i-1,1,k]-2*T[i,1,k] + T[i+1,1,k]) +
                               Gy*((-T[i,1,k]) + T[i,2,k]) +
                               Gz*(T[i,1,k-1]-2*T[i,1,k] + T[i,1,k + 1])) "Left face";
    
          der(T[i,M,k]) = 1/C*(Gx*(T[i-1,M,k]-2*T[i,M,k] + T[i+1,M,k]) +
                               Gy*(T[i,M-1,k]-T[i,M,k]) +
                               Gz*(T[i,M,k-1]-2*T[i,M,k] + T[i,M,k + 1])) "Right face";
        end for;
      end for;
    
      for j in 2:M-1 loop
        for k in 2:P-1 loop
          der(T[1,j,k]) = 1/C*(Gx*((-T[1,j,k]) + T[2,j,k]) +
                               Gy*(T[1,j-1,k]-2*T[1,j,k] + T[1,j+1,k]) +
                               Gz*(T[1,j,k-1]-2*T[1,j,k] + T[1,j,k + 1])) "Upper face";
          der(T[N,j,k]) = 1/C*(Gx*(T[N-1,j,k]-T[N,j,k]) +
                               Gy*(T[N,j-1,k]-2*T[N,j,k] + T[N,j+1,k]) +
                               Gz*(T[N,j,k-1]-2*T[N,j,k] + T[N,j,k + 1])) "Lower face";
        end for;
      end for;
    
      for i in 2:N-1 loop
        for j in 2:M-1 loop
          for k in 2:P-1 loop
            der(T[i,j,k]) = 1/C*(Gx*(T[i-1,j,k]-2*T[i,j,k] + T[i+1,j,k]) +
                                 Gy*(T[i,j-1,k]-2*T[i,j,k] + T[i,j+1,k]) +
                                 Gz*(T[i,j,k-1]-2*T[i,j,k] + T[i,j,k + 1])) "Internal volume";
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
      Types.Power Qvb[N,M,4,4] "Power dissipated in each area of each core mapped on the bottom surface volumes";
      Types.Power Qtot "Total dissipated power";
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
          Qvb[:,
              1:(TLC[i,2]+TLA[j,2])*Nr,
              i,j] = 
                zeros(12*Nr,(TLC[i,2]+TLA[j,2])*Nr);
          Qvb[1:(TLC[i,1]+TLA[j,1])*Nr,
              (TLC[i,2]+TLA[j,2])*Nr+1:(TLC[i,2]+TLA[j,2]+2)*Nr,
              i,j] = zeros((TLC[i,1]+TLA[j,1])*Nr,2*Nr);
          Qvb[(TLC[i,1]+TLA[j,1])*Nr+1:(TLC[i,1]+TLA[j,1]+2)*Nr,
              (TLC[i,2]+TLA[j,2])*Nr+1:(TLC[i,2]+TLA[j,2]+2)*Nr,
              i,j] = fill(Qv[i,j],2*Nr,2*Nr);
          Qvb[(TLC[i,1]+TLA[j,1]+2)*Nr+1:12*Nr,
              (TLC[i,2]+TLA[j,2])*Nr+1:(TLC[i,2]+TLA[j,2]+2)*Nr,
              i,j] = zeros(10-(TLC[i,1]+TLA[j,1])*Nr,2*Nr);
          Qvb[:,
              (TLC[i,2]+TLA[j,2]+2)*Nr+1:end,
              i,j] = zeros(12*Nr,10-(TLC[i,2]+TLA[j,2])*Nr);
        end for;
      end for;
      for i in 1:N loop
        for j in 1:M loop
          Qb[i,j] = sum(sum(Qvb[i,j,k,l] for k in 1:4) for l in 1:4);
        end for;
      end for;
      Qtot = sum(sum(Qb));
    annotation(
        experiment(StartTime = 0, StopTime = 4, Tolerance = 1e-6, Interval = 0.001),
        __OpenModelica_simulationFlags(lv = "LOG_STATS", outputFormat = "mat", s = "euler"));
    end ThermalChip4Cores;

    model ThermalChipSimpleBoundary "Thermal chip model written by explicit ODEs, constant power on half of the bottom surface"
      extends BaseThermalChip;
      parameter Types.Power Ptot = 100 "Total power consumption";
      final parameter Types.Power Pv = Ptot / (N * M / 2) "Power dissipated in a single volume";
    equation
      Qb[:, 1:div(M, 2)] = fill(Pv, N, div(M, 2));
      Qb[:, div(M, 2) + 1:end] = fill(0, N, M - div(M, 2));
      annotation(
        experiment(StartTime = 0, StopTime = 0.01, Tolerance = 1e-6, Interval = 0.000002),
        __OpenModelica_simulationFlags(lv = "LOG_STATS", outputFormat = "mat", s = "euler"));
    end ThermalChipSimpleBoundary;
  end Models;

	Models.ThermalChipSimpleBoundary  a;

  package Benchmarks
  end Benchmarks;
end ThermalChipODE;
