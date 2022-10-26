// RUN: marco --omc-bypass --model=TimeUsage --solver=forward-euler -o %basename_t %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x","y"
// CHECK-NEXT: 0.000000,0.000000,0.000000
// CHECK-NEXT: 0.100000,-0.544021,0.000000
// CHECK-NEXT: 0.200000,0.912945,-0.054402
// CHECK-NEXT: 0.300000,-0.988032,0.036892
// CHECK-NEXT: 0.400000,0.745113,-0.061911
// CHECK-NEXT: 0.500000,-0.262375,0.012601
// CHECK-NEXT: 0.600000,-0.304811,-0.013637
// CHECK-NEXT: 0.700000,0.773891,-0.044118
// CHECK-NEXT: 0.800000,-0.993889,0.033271
// CHECK-NEXT: 0.900000,0.893997,-0.066118
// CHECK-NEXT: 1.000000,-0.506366,0.023282

model TimeUsage
	Real x;
	Real y(start = 0, fixed = true);
equation
    x = sin(time * 100);
    der(y) = x;
end TimeUsage;
