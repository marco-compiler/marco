// RUN: marco --omc-bypass --model=CycleWithAllImplicitEquations --solver=ida -o %basename_t %s
// RUN: ./%basename_t --end-time=0.2 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x","y"
// CHECK-NEXT: 0.0000,0.7854,0.7854
// CHECK-NEXT: 0.1000,0.7854,0.7854
// CHECK-NEXT: 0.2000,0.7854,0.7854

model CycleWithAllImplicitEquations
	Real x(start = 0, fixed = false);
	Real y(start = 0, fixed = false);
equation
	x + cos(x) = y + sin(y);
	x + x*x = y + y*y;
end CycleWithAllImplicitEquations;
