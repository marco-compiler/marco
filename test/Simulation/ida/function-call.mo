// RUN: marco --omc-bypass --model=FunctionCall --solver=ida -o %basename_t %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x","y"
// CHECK-NEXT: 0.0000,0.0000,0.0000
// CHECK-NEXT: 0.1000,0.2000,0.0200
// CHECK-NEXT: 0.2000,0.4000,0.0795
// CHECK-NEXT: 0.3000,0.6000,0.1773
// CHECK-NEXT: 0.4000,0.8000,0.3116
// CHECK-NEXT: 0.5000,1.0000,0.4798
// CHECK-NEXT: 0.6000,1.2000,0.6788
// CHECK-NEXT: 0.7000,1.4000,0.9050
// CHECK-NEXT: 0.8000,1.6000,1.1546
// CHECK-NEXT: 0.9000,1.8000,1.4236
// CHECK-NEXT: 1.0000,2.0000,1.7081

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
