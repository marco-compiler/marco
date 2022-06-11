// RUN: marco --omc-bypass --model=ScalarVariablesSubstitution --end-time=1 --time-step=0.1 --solver=ida --ida-equidistant-time-grid -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: "time","x","y"
// CHECK-NEXT: 0.000000,0.000000,2.000000
// CHECK-NEXT: 0.100000,0.200000,2.000000
// CHECK-NEXT: 0.200000,0.400000,2.000000
// CHECK-NEXT: 0.300000,0.600000,2.000000
// CHECK-NEXT: 0.400000,0.800000,2.000000
// CHECK-NEXT: 0.500000,1.000000,2.000000
// CHECK-NEXT: 0.600000,1.200000,2.000000
// CHECK-NEXT: 0.700000,1.400000,2.000000
// CHECK-NEXT: 0.800000,1.600000,2.000000
// CHECK-NEXT: 0.900000,1.800000,2.000000
// CHECK-NEXT: 1.000000,2.000000,2.000000

model ScalarVariablesSubstitution
	Real x(start = 0, fixed = true);
	Real y;
equation
	der(x) = y;
	y = 2.0;
end ScalarVariablesSubstitution;
