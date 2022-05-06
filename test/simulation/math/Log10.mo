// RUN: marco --omc-bypass --model=Log10 --end-time=1 -o simulation_log10 %s
// RUN: ./simulation_log10 | FileCheck %s

// CHECK: time;x
// CHECK-NEXT: 0.000000000000;0.000000000000
// CHECK-NEXT: 0.100000000000;0.000000000000
// CHECK-NEXT: 0.200000000000;-inf
// CHECK-NEXT: 0.300000000000;nan
// CHECK-NEXT: 0.400000000000;nan
// CHECK-NEXT: 0.500000000000;nan
// CHECK-NEXT: 0.600000000000;nan
// CHECK-NEXT: 0.700000000000;nan
// CHECK-NEXT: 0.800000000000;nan
// CHECK-NEXT: 0.900000000000;nan
// CHECK-NEXT: 1.000000000000;nan

model Log10
    Real x;
equation
    der(x) = log10(x);
end Log10;
