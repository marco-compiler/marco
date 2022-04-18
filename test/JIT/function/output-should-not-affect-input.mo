// RUN: marco --omc-bypass --model=M1 --end-time=0.2 --time-step=0.1 -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: time;x[1];x[2];x[3];x[4];x[5];y[1];y[2];y[3];y[4];y[5]
// CHECK-NEXT: 0.000000;1;2;3;4;5;10;2;3;4;5
// CHECK-NEXT: 0.100000;1;2;3;4;5;10;2;3;4;5
// CHECK-NEXT: 0.200000;1;2;3;4;5;10;2;3;4;5

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
