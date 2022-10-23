// RUN: marco --omc-bypass --model=Model --end-time=0.2 --time-step=0.1 -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: "time","x[1]","x[2]","x[3]","y[1]","y[2]"
// CHECK-NEXT: 0.000000,2.500000,1.000000,-1.000000,0.000000,-0.500000
// CHECK-NEXT: 0.100000,2.500000,1.000000,-1.000000,0.000000,-0.500000
// CHECK-NEXT: 0.200000,2.500000,1.000000,-1.000000,0.000000,-0.500000

model Model
    Real[3] x;
    Real[2] y;
equation
    x[1] + x[3] + y[2] = 1;
    x[1] + x[3] - y[2] = 2;
    x[1] - x[3] + y[2] = 3;
    x[2] + y[1] = 1;
    x[2] - y[1] = 1;
end Model;
