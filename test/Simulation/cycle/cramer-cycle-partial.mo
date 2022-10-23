// RUN: marco --omc-bypass --model=Model --end-time=0.2 --time-step=0.1 -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: "time","a","b","x","y","z"
// CHECK-NEXT: 0.000000,1.000000,0.000000,2.500000,-1.000000,-0.500000
// CHECK-NEXT: 0.100000,1.000000,0.000000,2.500000,-1.000000,-0.500000
// CHECK-NEXT: 0.200000,1.000000,0.000000,2.500000,-1.000000,-0.500000

model Model
    Real a;
    Real b;
    Real x;
    Real y;
    Real z;
equation
    a + b = 1;
    a - b = 1;
    x + y + z = 1;
    x + y - z = 2;
    x - y + z = 3;
end Model;
