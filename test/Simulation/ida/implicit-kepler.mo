// RUN: marco --omc-bypass --model=ImplicitKepler --end-time=1 --time-step=0.1 --solver=ida --ida-equidistant-time-grid -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: "time","x[1]","x[2]"
// CHECK-NEXT: 0.000000,3.657695,3.657695
// CHECK-NEXT: 0.100000,3.657695,3.657695
// CHECK-NEXT: 0.200000,3.657695,3.657695
// CHECK-NEXT: 0.300000,3.657695,3.657695
// CHECK-NEXT: 0.400000,3.657695,3.657695
// CHECK-NEXT: 0.500000,3.657695,3.657695
// CHECK-NEXT: 0.600000,3.657695,3.657695
// CHECK-NEXT: 0.700000,3.657695,3.657695
// CHECK-NEXT: 0.800000,3.657695,3.657695
// CHECK-NEXT: 0.900000,3.657695,3.657695
// CHECK-NEXT: 1.000000,3.657695,3.657695

model ImplicitKepler
	Real[2] x(each start = 3.6, fixed = false);
equation
	for i in 1:2 loop
		5.0 = x[i] - 2.72 * sin(x[i]);
	end for;
end ImplicitKepler;
