// RUN: marco --omc-bypass --model=Model --end-time=0.2 --time-step=0.1 -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: "time","a","b","c","x","y","z"
// CHECK-NEXT: 0.000000,2.500000,-2.250000,-1.750000,2.500000,-1.000000,-0.500000
// CHECK-NEXT: 0.100000,2.500000,-2.250000,-1.750000,2.500000,-1.000000,-0.500000
// CHECK-NEXT: 0.200000,2.500000,-2.250000,-1.750000,2.500000,-1.000000,-0.500000

model Model
    Real a;
    Real b;
    Real c;
    Real x;
    Real y;
    Real z;
equation
    a + b + c = 1;
    a + b - c = 2;
    a - b + c = 3;
    x + y + z + b - 2*c = 1;
    x + y - z = 2;
    x - y + z = 3;
end Model;
