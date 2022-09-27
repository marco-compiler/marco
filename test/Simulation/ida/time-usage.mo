// RUN: marco --omc-bypass --model=TimeUsage --end-time=1 --variable-filter="time;x;der(x)" --time-step=0.1 --solver=ida --ida-equidistant-time-grid -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: "time","x"
// CHECK-NEXT: 0.000000,0.000000,0.000000
// CHECK-NEXT: 0.100000,0.005000,0.100000
// CHECK-NEXT: 0.200000,0.020000,0.200000
// CHECK-NEXT: 0.300000,0.045000,0.300000
// CHECK-NEXT: 0.400000,0.080000,0.400000
// CHECK-NEXT: 0.500000,0.125000,0.500000
// CHECK-NEXT: 0.600000,0.180000,0.600000
// CHECK-NEXT: 0.700000,0.245000,0.700000
// CHECK-NEXT: 0.800000,0.320000,0.800000
// CHECK-NEXT: 0.900000,0.405000,0.900000
// CHECK-NEXT: 1.000000,0.500000,1.000000

model TimeUsage
	Real x(start = 0, fixed = true);
equation
	der(x) = time;
end TimeUsage;
