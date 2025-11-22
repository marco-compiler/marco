// RUN: cc -c %S/scalar-arguments-and-result.c -o %basename_t_lib.o
// RUN: marco %s %basename_t_lib.o --omc-bypass --model=Test --solver=euler-forward -o %basename_t %link_flags -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.5 --precision=6 | FileCheck %s

// CHECK: "time","x1","x2","y1","y2","z1","z2"
// CHECK: 0.000000,1.000000,1.200000,2.000000,2.300000,3.000000,3.500000
// CHECK: 1.000000,1.000000,1.200000,2.000000,2.300000,3.000000,3.500000

function integerSum
	input Integer x;
	input Integer y;
	output Integer z;
external z = sumInteger(x, y);
end integerSum;

function realSum
	input Real x;
	input Real y;
	output Real z;
external z = sumReal(x, y);
end realSum;

model Test
	Real x1 = 1;
	Real y1 = 2;
	Real z1;

	Real x2 = 1.2;
	Real y2 = 2.3;
	Real z2;
equation
    z1 = integerSum(x1, y1);
    z2 = realSum(x2, y2);
end Test;
