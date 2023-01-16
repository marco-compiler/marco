// RUN: marco --omc-bypass --model=InlineIf --solver=euler-forward -o %basename_t %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x"
// CHECK-NEXT: 0.000000,0.000000
// CHECK-NEXT: 0.100000,0.000000
// CHECK-NEXT: 0.200000,0.000000
// CHECK-NEXT: 0.300000,0.000000
// CHECK-NEXT: 0.400000,0.000000
// CHECK-NEXT: 0.500000,0.000000
// CHECK-NEXT: 0.600000,1.000000
// CHECK-NEXT: 0.700000,1.000000
// CHECK-NEXT: 0.800000,1.000000
// CHECK-NEXT: 0.900000,1.000000
// CHECK-NEXT: 1.000000,1.000000

model InlineIf
	Real x(each start = 0, fixed = false);
equation
   x = if time > 0.5 then 1 else 0;
end InlineIf;
