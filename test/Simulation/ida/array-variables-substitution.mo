// RUN: marco --omc-bypass --model=ArrayVariablesSubstitution --solver=ida -o %basename_t %s -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=2 | FileCheck %s

// CHECK: "time","Q[1]","Q[2]","Q[3]","Q[4]","Q[5]","T[1]","T[2]","T[3]","T[4]"
// CHECK-NEXT: 0.00,10.00,0.00,0.00,0.00,10.00,100.00,100.00,100.00,100.00
// CHECK-NEXT: 0.10,10.00,0.91,0.09,0.91,10.00,100.95,100.04,99.96,99.05
// CHECK-NEXT: 0.20,10.00,1.67,0.31,1.67,10.00,101.82,100.15,99.85,98.18
// CHECK-NEXT: 0.30,10.00,2.31,0.62,2.31,10.00,102.62,100.31,99.69,97.38
// CHECK-NEXT: 0.40,10.00,2.87,0.98,2.87,10.00,103.36,100.49,99.51,96.64
// CHECK-NEXT: 0.50,10.00,3.37,1.37,3.37,10.00,104.05,100.68,99.32,95.95
// CHECK-NEXT: 0.60,10.00,3.81,1.77,3.81,10.00,104.69,100.89,99.11,95.31
// CHECK-NEXT: 0.70,10.00,4.20,2.18,4.20,10.00,105.29,101.09,98.91,94.71
// CHECK-NEXT: 0.80,10.00,4.56,2.58,4.56,10.00,105.85,101.29,98.71,94.15
// CHECK-NEXT: 0.90,10.00,4.89,2.97,4.89,10.00,106.38,101.49,98.51,93.62
// CHECK-NEXT: 1.00,10.00,5.20,3.35,5.20,10.00,106.87,101.67,98.33,93.13

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
