// RUN: marco --omc-bypass --model=Model --end-time=0.2 --time-step=0.1 -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: "time","x","y","z","q","r"
// CHECK-NEXT: 0.000000,2.500000,-1.000000,-0.500000,1.000000,0.000000
// CHECK-NEXT: 0.100000,2.500000,-1.000000,-0.500000,1.000000,0.000000
// CHECK-NEXT: 0.200000,2.500000,-1.000000,-0.500000,1.000000,0.000000

model Model
    Real x;
    Real y;
    Real z;
    Real q;
    Real r;
equation
    x + y + z = 1;
    x + y - z = 2;
    x - y + z = 3;
    q + r = 1;
    q - r = 1;
end Model;
