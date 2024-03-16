// RUN: marco --omc-bypass --model=ArrayVariablesSubstitution --solver=ida -o %basename_t %s -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=2 | FileCheck %s

// CHECK: "time","Q[1]","Q[2]","Q[3]","Q[4]","Q[5]","T[1]","T[2]","T[3]","T[4]"
// CHECK: 0.00,10.00,0.00,0.00,0.00,10.00,100.00,100.00,100.00,100.00
// CHECK: 1.00,10.00,5.20,3.35,5.20,10.00,106.87,101.67,98.33,93.13

model ArrayVariablesSubstitution
  Real[4] T(each start = 100.0, fixed = true);
  Real[5] Q;
equation
  for i in 1:4 loop
    der(T[i]) = Q[i] - Q[i + 1];
  end for;

  Q[1] = 10;

  for i in 2:4 loop
    Q[i] = T[i - 1] - T[i];
  end for;

  Q[5] = 10;
end ArrayVariablesSubstitution;
