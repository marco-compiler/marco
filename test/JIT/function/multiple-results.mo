// RUN: marco --omc-bypass --model=M1 --end-time=0.2 --time-step=0.1 -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: "time","x","y[1]","y[2]"
// CHECK-NEXT: 0.000000,2,4,6
// CHECK-NEXT: 0.100000,2,4,6
// CHECK-NEXT: 0.200000,2,4,6

function bar
    input Integer x;
    output Integer y;
    output Integer z;

algorithm
    y := 2 * x;
    z := 3 * x;
end bar;

function foo
    input Integer x;
    output Integer[2] y;

algorithm
    (y[1], y[2]) := bar(x);
end foo;

model M1
    Integer x;
    Integer[2] y;
equation
    x = 2;
    y = foo(x);
end M1;
