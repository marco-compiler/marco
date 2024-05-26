// RUN: marco %s --omc-bypass --model=FunctionCall --solver=ida -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir -L %llvm_lib_dir -Wl,-rpath,%llvm_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x","y"
// CHECK: 0.0000,0.0000,0.0000
// CHECK: 1.0000,2.0000,1.7081

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
