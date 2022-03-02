// RUN: marco --omc-bypass --model=M1 --end-time=0.2 --time-step=0.1 -o %basename_t %s
// RUN: ./%basename_t | FileCheck %s

// CHECK: time;x;y;z
// CHECK-NEXT: 0.000000000000;0.000000000000;0.000000000000;0.000000000000
// CHECK-NEXT: 0.100000000000;-0.500000000000;3.500000000000;-1.500000000000
// CHECK-NEXT: 0.200000000000;-0.500000000000;3.500000000000;-1.500000000000

model M1
    Real x;
    Real y;
    Real z;
equation
    x = y - 4;
    y = 2 - z;
    z = 3 * x;
end M1;
