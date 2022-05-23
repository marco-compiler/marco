// RUN: marco --omc-bypass --model=M1 --end-time=0.2 --time-step=0.1 -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: "time","x[1]","x[2]","x[3]","y[1]","y[2]","y[3]","z[1]","z[2]","z[3]"
// CHECK-NEXT: 0.000000,-0.500000,-0.500000,-0.500000,3.500000,3.500000,3.500000,-1.500000,-1.500000,-1.500000
// CHECK-NEXT: 0.100000,-0.500000,-0.500000,-0.500000,3.500000,3.500000,3.500000,-1.500000,-1.500000,-1.500000
// CHECK-NEXT: 0.200000,-0.500000,-0.500000,-0.500000,3.500000,3.500000,3.500000,-1.500000,-1.500000,-1.500000

model M1
    Real[3] x;
    Real[3] y;
    Real[3] z;
equation
    for i in 1:3 loop
        x[i] = y[i] - 4;
    end for;

    for i in 1:3 loop
        y[i] = 2 - z[i];
    end for;

    for i in 1:3 loop
        z[i] = 3 * x[i];
    end for;
end M1;
