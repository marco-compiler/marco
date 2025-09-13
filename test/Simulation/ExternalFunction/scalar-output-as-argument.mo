// RUN: cc -c %S/Library.c -o %basename_t.o
// RUN: marco %s %basename_t.o --omc-bypass --model=Test --solver=euler-forward -o %basename_t -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=2 --time-step=1 --precision=6 | FileCheck %s

// CHECK: "time","x","y","z"
// CHECK: 0.000000,1.000000,2.000000,3.000000
// CHECK: 1.000000,1.000000,2.000000,3.000000
// CHECK: 2.000000,1.000000,2.000000,3.000000

function bar
	input Integer x;
	input Integer[2] y;
	output Integer z;
	output Integer[2] t;
external sum(x, y, z, t);
end bar;

function foo
	Integer x;
	Integer[2] y;
	Integer z;
	Integer[2] t;
algorithm
    (z, t) := bar(x, y);
end foo;

model Test
end Test;
