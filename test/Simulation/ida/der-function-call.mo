// RUN: marco %s --omc-bypass --model=DerFunctionCall --solver=ida -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir -L %llvm_lib_dir -Wl,-rpath,%llvm_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","w","z"
// CHECK: 0.0000,0.0000,0.0000
// CHECK: 1.0000,159.9600,53.3200

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
