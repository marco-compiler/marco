// RUN: marco --omc-bypass --model=CycleWithAllImplicitEquations --solver=ida -o %basename_t %s
// RUN: ./%basename_t --end-time=0.2 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x","y"
// CHECK-NEXT: 0.000000,0.785398,0.785398
// CHECK-NEXT: 0.100000,0.785398,0.785398
// CHECK-NEXT: 0.200000,0.785398,0.785398

model CycleWithAllImplicitEquations
	Real x(start = 0, fixed = false);
	Real y(start = 0, fixed = false);
equation
	x + cos(x) = y + sin(y);
	x + x*x = y + y*y;
end CycleWithAllImplicitEquations;
