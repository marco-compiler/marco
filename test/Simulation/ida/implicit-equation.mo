// RUN: marco --omc-bypass --model=ImplicitEquation --end-time=1 --time-step=0.1 --solver=ida --ida-equidistant-time-grid -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: "time","x"
// CHECK-NEXT: 0.000000,1.739204
// CHECK-NEXT: 0.100000,1.739204
// CHECK-NEXT: 0.200000,1.739204
// CHECK-NEXT: 0.300000,1.739204
// CHECK-NEXT: 0.400000,1.739204
// CHECK-NEXT: 0.500000,1.739204
// CHECK-NEXT: 0.600000,1.739204
// CHECK-NEXT: 0.700000,1.739204
// CHECK-NEXT: 0.800000,1.739204
// CHECK-NEXT: 0.900000,1.739204
// CHECK-NEXT: 1.000000,1.739204

model ImplicitEquation
	Real x(start = 0, fixed = false);
equation
	x + x * x * x = 7.0;
end ImplicitEquation;
