// RUN: marco %s --omc-bypass --model=ScalarVariablesSubstitution --solver=ida -o %basename_t -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x","y"
// CHECK: 0.0000,0.0000,2.0000
// CHECK: 1.0000,2.0000,2.0000

model ScalarVariablesSubstitution
	Real x(start = 0, fixed = true);
	Real y;
equation
	der(x) = y;
	y = 2.0;
end ScalarVariablesSubstitution;
