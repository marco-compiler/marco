// RUN: marco --omc-bypass --model=DerFunctionCall --solver=ida -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","w","z"
// CHECK-NEXT: 0.0000,0.0000,0.0000
// CHECK-NEXT: 0.1000,15.9960,5.3320
// CHECK-NEXT: 0.2000,31.9920,10.6640
// CHECK-NEXT: 0.3000,47.9880,15.9960
// CHECK-NEXT: 0.4000,63.9840,21.3280
// CHECK-NEXT: 0.5000,79.9800,26.6600
// CHECK-NEXT: 0.6000,95.9760,31.9920
// CHECK-NEXT: 0.7000,111.9720,37.3240
// CHECK-NEXT: 0.8000,127.9680,42.6560
// CHECK-NEXT: 0.9000,143.9640,47.9880
// CHECK-NEXT: 1.0000,159.9600,53.3200

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
