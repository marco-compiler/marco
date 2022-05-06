// RUN: marco --omc-bypass --model=Atan --end-time=1 -o simulation_atan %s
// RUN: ./simulation_atan | FileCheck %s

// CHECK: time;x
// CHECK-NEXT: 0.000000000000;0.000000000000
// CHECK-NEXT: 0.100000000000;0.000000000000
// CHECK-NEXT: 0.200000000000;0.100000000000
// CHECK-NEXT: 0.300000000000;0.190033134751
// CHECK-NEXT: 0.400000000000;0.271253742093
// CHECK-NEXT: 0.500000000000;0.344765740110
// CHECK-NEXT: 0.600000000000;0.411565320833
// CHECK-NEXT: 0.700000000000;0.472521665671
// CHECK-NEXT: 0.800000000000;0.528379235999
// CHECK-NEXT: 0.900000000000;0.579769996114
// CHECK-NEXT: 1.000000000000;0.627228829183

model Atan
    Real x;
equation
    der(x) = 1 - atan(x);
end Atan;
