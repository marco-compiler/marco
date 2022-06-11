// RUN: marco --omc-bypass --model=CycleWithImplicitEquation --end-time=1 --time-step=0.1 --solver=ida --ida-equidistant-time-grid -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: "time","x","y"
// CHECK-NEXT: 0.000000,-0.664044,-1.187815
// CHECK-NEXT: 0.100000,-0.664044,-1.187815
// CHECK-NEXT: 0.200000,-0.664044,-1.187815
// CHECK-NEXT: 0.300000,-0.664044,-1.187815
// CHECK-NEXT: 0.400000,-0.664044,-1.187815
// CHECK-NEXT: 0.500000,-0.664044,-1.187815
// CHECK-NEXT: 0.600000,-0.664044,-1.187815
// CHECK-NEXT: 0.700000,-0.664044,-1.187815
// CHECK-NEXT: 0.800000,-0.664044,-1.187815
// CHECK-NEXT: 0.900000,-0.664044,-1.187815
// CHECK-NEXT: 1.000000,-0.664044,-1.187815

model CycleWithImplicitEquation
    Real x(start = -0.7, fixed = false);
    Real y(start = -0.7, fixed = false);
equation
    x * x + y * y + x + y = 0;
    x * x * y + x = y;
end CycleWithImplicitEquation;
