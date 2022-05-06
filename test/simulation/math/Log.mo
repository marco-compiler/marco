// RUN: marco --omc-bypass --model=Log --end-time=1 -o simulation_log %s
// RUN: ./simulation_log | FileCheck %s

// CHECK: time;x
// CHECK-NEXT: 0.000000000000;0.000000000000
// CHECK-NEXT: 0.100000000000;0.000000000000
// CHECK-NEXT: 0.200000000000;-inf
// CHECK-NEXT: 0.300000000000;-nan
// CHECK-NEXT: 0.400000000000;-nan
// CHECK-NEXT: 0.500000000000;-nan
// CHECK-NEXT: 0.600000000000;-nan
// CHECK-NEXT: 0.700000000000;-nan
// CHECK-NEXT: 0.800000000000;-nan
// CHECK-NEXT: 0.900000000000;-nan
// CHECK-NEXT: 1.000000000000;-nan

model Log
    Real x;
equation
    der(x) = log(x);
end Log;
