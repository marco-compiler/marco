// RUN: marco --omc-bypass --model=Model --end-time=0.2 --time-step=0.1 -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: "time","x[1]","x[2]","x[3]"
// CHECK-NEXT: 0.000000,-1.000000,2.000000,-1.000000
// CHECK-NEXT: 0.100000,-1.000000,2.000000,-1.000000
// CHECK-NEXT: 0.200000,-1.000000,2.000000,-1.000000

model Model
    Real[3] x;
equation
    for i in 1:2 loop
        x[i] + x[i+1] = 1;
    end for;
    x[1] + x[2] + x[3] = 0;
end Model;
