// RUN: marco %s --modelica-frontend=native --model=Circuit.Ladder --solver=ida -o %basename_t %link_flags -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.5 --precision=6 | FileCheck %s

// Model flattened by the native frontend, with arrays of components connected
// within for-loops. The variables of the arrays of components are kept as
// arrays.

// CHECK: "time","N","c.C[1]","c.C[2]","c.i[1]","c.i[2]","c.n.i[1]","c.n.i[2]","c.n.v[1]","c.n.v[2]","c.p.i[1]","c.p.i[2]","c.p.v[1]","c.p.v[2]","c.v[1]","c.v[2]","g.p.i","g.p.v","r.R[1]","r.R[2]","r.i[1]","r.i[2]","r.n.i[1]","r.n.i[2]","r.n.v[1]","r.n.v[2]","r.p.i[1]","r.p.i[2]","r.p.v[1]","r.p.v[2]","r.v[1]","r.v[2]","s.i","s.n.i","s.n.v","s.p.i","s.p.v","s.v"
// CHECK-NEXT: 0.000000,2.000000,0.500000,0.500000,1.000000,0.000000,-1.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000,2.000000,1.000000,0.000000,-1.000000,0.000000,0.000000,0.000000,1.000000,0.000000,1.000000,0.000000,1.000000,0.000000,-1.000000,1.000000,0.000000,-1.000000,1.000000,1.000000
// CHECK-NEXT: 0.500000,2.000000,0.500000,0.500000,0.264086,0.199657,-0.264086,-0.199657,0.000000,0.000000,0.264086,0.199657,0.536257,0.136942,0.536257,0.136942,0.000000,0.000000,1.000000,2.000000,0.463743,0.199657,-0.463743,-0.199657,0.536257,0.136942,0.463743,0.199657,1.000000,0.536257,0.463743,0.399315,-0.463743,0.463743,0.000000,-0.463743,1.000000,1.000000
// CHECK-NEXT: 1.000000,2.000000,0.500000,0.500000,0.109605,0.185179,-0.109605,-0.185179,0.000000,0.000000,0.109605,0.185179,0.705215,0.334857,0.705215,0.334857,0.000000,0.000000,1.000000,2.000000,0.294785,0.185179,-0.294785,-0.185179,0.705215,0.334857,0.294785,0.185179,1.000000,0.705215,0.294785,0.370359,-0.294785,0.294785,0.000000,-0.294785,1.000000,1.000000

package Circuit
  connector Pin
    Real v;
    flow Real i;
  end Pin;

  partial model OnePort
    Pin p, n;
    Real v;
    Real i;
  equation
    v = p.v - n.v;
    0 = p.i + n.i;
    i = p.i;
  end OnePort;

  model Resistor
    extends OnePort;
    parameter Real R = 1;
  equation
    v = R * i;
  end Resistor;

  model Capacitor
    extends OnePort(v(start = 0, fixed = true));
    parameter Real C = 1;
  equation
    C * der(v) = i;
  end Capacitor;

  model Source
    extends OnePort;
  equation
    v = 1;
  end Source;

  model Ground
    Pin p;
  equation
    p.v = 0;
  end Ground;

  model Ladder
    parameter Integer N = 2;
    Source s;
    Ground g;
    Resistor r[N](R = {1, 2});
    Capacitor c[N](each C = 0.5);
  equation
    connect(s.p, r[1].p);
    for k in 1:N - 1 loop
      connect(r[k].n, r[k + 1].p);
    end for;
    for k in 1:N loop
      connect(r[k].n, c[k].p);
      connect(c[k].n, g.p);
    end for;
    connect(s.n, g.p);
  end Ladder;
end Circuit;
