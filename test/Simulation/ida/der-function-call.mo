// RUN: marco --omc-bypass --model=DerFunctionCall --solver=ida -o %basename_t %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","w","z"
// CHECK-NEXT: 0.000000,0.000000,0.000000
// CHECK-NEXT: 0.100000,15.996000,5.332000
// CHECK-NEXT: 0.200000,31.992000,10.664000
// CHECK-NEXT: 0.300000,47.988000,15.996000
// CHECK-NEXT: 0.400000,63.984000,21.328000
// CHECK-NEXT: 0.500000,79.980000,26.660000
// CHECK-NEXT: 0.600000,95.976000,31.992000
// CHECK-NEXT: 0.700000,111.972000,37.324000
// CHECK-NEXT: 0.800000,127.968000,42.656000
// CHECK-NEXT: 0.900000,143.964000,47.988000
// CHECK-NEXT: 1.000000,159.960000,53.320000

function foo
	input Real T;
	output Real cp;
	protected Real[2] u;
	protected Real x;
algorithm
	u := {4.5, 3.3};
	x := 5.2;

	for i in 1:2 loop
		x := x + {2.7, 10.9}[i] * u[i];
	end for;

	cp := x * T;
end foo;

function bar = der(foo, T);

model DerFunctionCall
	Real z(start = 0, fixed = true);
	Real w(start = 0, fixed = true);
equation
	der(w) = foo(3.0);
	der(z) = bar(3.0);
end DerFunctionCall;
