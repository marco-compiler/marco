// RUN: marco --omc-bypass --model=ScalarVariablesSubstitution --solver=ida -o %basename_t -L %runtime_lib_dir %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x","y"
// CHECK-NEXT: 0.0000,0.0000,2.0000
// CHECK-NEXT: 0.1000,0.2000,2.0000
// CHECK-NEXT: 0.2000,0.4000,2.0000
// CHECK-NEXT: 0.3000,0.6000,2.0000
// CHECK-NEXT: 0.4000,0.8000,2.0000
// CHECK-NEXT: 0.5000,1.0000,2.0000
// CHECK-NEXT: 0.6000,1.2000,2.0000
// CHECK-NEXT: 0.7000,1.4000,2.0000
// CHECK-NEXT: 0.8000,1.6000,2.0000
// CHECK-NEXT: 0.9000,1.8000,2.0000
// CHECK-NEXT: 1.0000,2.0000,2.0000

model ScalarVariablesSubstitution
	Real x(start = 0, fixed = true);
	Real y;
equation
	der(x) = y;
	y = 2.0;
end ScalarVariablesSubstitution;
