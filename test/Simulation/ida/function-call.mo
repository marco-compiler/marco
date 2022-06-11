// RUN: marco --omc-bypass --model=FunctionCall --end-time=1 --time-step=0.1 --solver=ida --ida-equidistant-time-grid -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: "time","x","y"
// CHECK-NEXT: 0.000000,0.000000,0.000000
// CHECK-NEXT: 0.100000,0.200000,0.019963
// CHECK-NEXT: 0.200000,0.400000,0.079460
// CHECK-NEXT: 0.300000,0.600000,0.177320
// CHECK-NEXT: 0.400000,0.800000,0.311635
// CHECK-NEXT: 0.500000,1.000000,0.479838
// CHECK-NEXT: 0.600000,1.200000,0.678811
// CHECK-NEXT: 0.700000,1.400000,0.905007
// CHECK-NEXT: 0.800000,1.600000,1.154591
// CHECK-NEXT: 0.900000,1.800000,1.423592
// CHECK-NEXT: 1.000000,2.000000,1.708064

function foo
	input Real x;
	output Real y;
algorithm
	y := x + sin(x);
end foo;

model FunctionCall
	Real x(start = 0, fixed = true);
	Real y(start = 0, fixed = true);
equation
	der(x) = 2;
	der(y) = foo(x);
end FunctionCall;
