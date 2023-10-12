// RUN: marco --omc-bypass --model=M1 --solver=euler-forward -o %basename_t -L %runtime_lib_dir %s
// RUN: ./%basename_t --precision=6 --end-time=0.2 --time-step=0.1 | FileCheck %s

// CHECK: "time","x[1]","x[2]","x[3]","x[4]","x[5]","y[1]","y[2]","y[3]","y[4]","y[5]"
// CHECK-NEXT: 0.000000,1.000000,2.000000,3.000000,4.000000,5.000000,10.000000,2.000000,3.000000,4.000000,5.000000
// CHECK-NEXT: 0.100000,1.000000,2.000000,3.000000,4.000000,5.000000,10.000000,2.000000,3.000000,4.000000,5.000000
// CHECK-NEXT: 0.200000,1.000000,2.000000,3.000000,4.000000,5.000000,10.000000,2.000000,3.000000,4.000000,5.000000

function foo
    input Integer[:] x;
    output Integer[:] y;

algorithm
    y := x;
    y[1] := 10;
end foo;

model M1
    Integer[5] x;
    Integer[5] y;
equation
    x[1] = 1;
    x[2] = 2;
    x[3] = 3;
    x[4] = 4;
    x[5] = 5;
    y = foo(x);
end M1;
