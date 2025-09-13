// RUN: cc -c %S/Library.c -o %basename_t.o
// RUN: marco %s %basename_t.o --omc-bypass --model=Test --solver=euler-forward -o %basename_t -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=2 --time-step=1 --precision=6 | FileCheck %s

// CHECK: "time","x","y","z"
// CHECK: 0.000000,1.000000,2.000000,3.000000
// CHECK: 1.000000,1.000000,2.000000,3.000000
// CHECK: 2.000000,1.000000,2.000000,3.000000

function foo
	input Real x;
	input Real y;
	output Real z;
external z = sum(x, y);
end foo;

model Test
	Real x = 1;
	Real y = 2;
	Real z;
equation
    z = foo(x, y);
end Test;
