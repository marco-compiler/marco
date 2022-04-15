// RUN: marco --omc-bypass --model=CycleWithAllImplicitEquations --end-time=1 --time-step=0.1 --solver=ida --ida-equidistant-time-grid -o %basename_t %s
// RUN: ./%basename_t | FileCheck %s

// CHECK: time;x;y
// CHECK-NEXT: 0.000000000000;1.739203861217
// CHECK-NEXT: 0.100000000000;1.739203861217
// CHECK-NEXT: 0.200000000000;1.739203861217
// CHECK-NEXT: 0.300000000000;1.739203861217
// CHECK-NEXT: 0.400000000000;1.739203861217
// CHECK-NEXT: 0.500000000000;1.739203861217
// CHECK-NEXT: 0.600000000000;1.739203861217
// CHECK-NEXT: 0.700000000000;1.739203861217
// CHECK-NEXT: 0.800000000000;1.739203861217
// CHECK-NEXT: 0.900000000000;1.739203861217
// CHECK-NEXT: 1.000000000000;1.739203861217

model CycleWithAllImplicitEquations
    Real x(start = -2);
    Real y(start = -2);
equation
    x * x + y * y + x + y = 0;
    x * x * y + x = y;
end CycleWithAllImplicitEquations;
