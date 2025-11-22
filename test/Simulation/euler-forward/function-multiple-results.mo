// RUN: marco %s --omc-bypass --model=M1 --solver=euler-forward -o %basename_t %link_flags -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --precision=6 --end-time=0.2 --time-step=0.1 | FileCheck %s

// CHECK: "time","x","y[1]","y[2]"
// CHECK: 0.000000,2.000000,4.000000,6.000000
// CHECK: 0.200000,2.000000,4.000000,6.000000

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
