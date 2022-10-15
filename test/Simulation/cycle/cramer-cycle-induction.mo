// RUN: marco --omc-bypass --model=Model --end-time=0.2 --time-step=0.1 -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: "time","x[1]","x[2]","x[3]","x[4]","x[5]","x[6]","x[7]"
// CHECK-NEXT: 0.000000,4.000000,-1.500000,1.000000,1.000000,-1.500000,-1.000000,-0.500000
// CHECK-NEXT: 0.100000,4.000000,-1.500000,1.000000,1.000000,-1.500000,-1.000000,-0.500000
// CHECK-NEXT: 0.200000,4.000000,-1.500000,1.000000,1.000000,-1.500000,-1.000000,-0.500000

model Model
    Real[7] x;
equation
    x[1] + x[2] + x[6] - x[7] = 2;
    x[1] + x[2] + x[6] + x[7] = 1;
    x[1] + x[2] - x[6] + x[7] = 3;
    x[1] - x[2] + x[6] + x[7] = 4;
    for i in 1:3 loop
        x[i] + x[i+1] + x[i+2] - x[i+3] + x[i+4] = 1;
    end for;
end Model;
