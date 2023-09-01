// RUN: marco --omc-bypass --model=M1 --solver=euler-forward -o %basename_t %s
// RUN: ./%basename_t --end-time=0.2 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x[1]","x[2]","x[3]","x[4]","y[1]","y[2]","y[3]","y[4]","z[1]","z[2]","z[3]","z[4]"
// CHECK-NEXT: 0.000000,3.000000,3.000000,2.000000,2.000000,5.000000,5.000000,4.000000,4.000000,5.000000,5.000000,4.000000,4.000000
// CHECK-NEXT: 0.100000,3.000000,3.000000,2.000000,2.000000,5.000000,5.000000,4.000000,4.000000,5.000000,5.000000,4.000000,4.000000
// CHECK-NEXT: 0.200000,3.000000,3.000000,2.000000,2.000000,5.000000,5.000000,4.000000,4.000000,5.000000,5.000000,4.000000,4.000000

model M1
    Real[4] x;
    Real[4] y;
    Real[4] z;
equation
    for i in 1:4 loop
        x[i] = y[i] - 2;
    end for;

    for i in 1:4 loop
        y[i] = z[i];
    end for;

    for i in 1:2 loop
        z[i] = 3 * x[i] - 4;
    end for;

    for i in 3:4 loop
        z[i] = 5 * x[i] - 6;
    end for;
end M1;
