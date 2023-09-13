// RUN: marco --omc-bypass --model=ImplicitEquation --solver=ida -o %basename_t %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x"
// CHECK-NEXT: 0.0000,1.7392
// CHECK-NEXT: 0.1000,1.7392
// CHECK-NEXT: 0.2000,1.7392
// CHECK-NEXT: 0.3000,1.7392
// CHECK-NEXT: 0.4000,1.7392
// CHECK-NEXT: 0.5000,1.7392
// CHECK-NEXT: 0.6000,1.7392
// CHECK-NEXT: 0.7000,1.7392
// CHECK-NEXT: 0.8000,1.7392
// CHECK-NEXT: 0.9000,1.7392
// CHECK-NEXT: 1.0000,1.7392

model ImplicitEquation
	Real x(start = 0, fixed = false);
equation
	x + x * x * x = 7.0;
end ImplicitEquation;
