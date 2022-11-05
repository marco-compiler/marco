// RUN: marco --omc-bypass --model=Model --end-time=0.2 --time-step=0.1 -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: "time","a","b","c","d"
// CHECK-NEXT: 0.000000,4.000000,-1.500000,-1.000000,-0.500000
// CHECK-NEXT: 0.100000,4.000000,-1.500000,-1.000000,-0.500000
// CHECK-NEXT: 0.200000,4.000000,-1.500000,-1.000000,-0.500000

model Model
    Real a;
    Real b;
    Real c;
    Real d;
equation
    a + b + c + d = 1;
    a + b + c - d = 2;
    a + b - c + d = 3;
    a - b + c + d = 4;
end Model;
