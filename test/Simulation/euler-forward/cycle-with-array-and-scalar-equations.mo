// RUN: marco --omc-bypass --model=ArrayAndScalarEquations --solver=euler-forward -o %basename_t -L %runtime_lib_dir %s
// RUN: ./%basename_t --end-time=0.2 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x[1]","x[2]","x[3]","x[4]","y"
// CHECK-NEXT: 0.000000,1.000000,1.000000,2.000000,2.000000,-5.000000
// CHECK-NEXT: 0.100000,1.000000,1.000000,2.000000,2.000000,-5.000000
// CHECK-NEXT: 0.200000,1.000000,1.000000,2.000000,2.000000,-5.000000

model ArrayAndScalarEquations
    Real[4] x;
    Real y;
equation
    for i in 1:2 loop
        x[i] + x[i+2] = 3;
    end for;

    x[3] = y + 7;

    for i in 3:4 loop
        x[i] - x[i-2] = 1;
    end for;
end ArrayAndScalarEquations;
