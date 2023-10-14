// RUN: marco --omc-bypass --model=CycleWithImplicitEquation --solver=ida -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x","y"
// CHECK-NEXT: 0.0000,-0.6640,-1.1878
// CHECK-NEXT: 0.1000,-0.6640,-1.1878
// CHECK-NEXT: 0.2000,-0.6640,-1.1878
// CHECK-NEXT: 0.3000,-0.6640,-1.1878
// CHECK-NEXT: 0.4000,-0.6640,-1.1878
// CHECK-NEXT: 0.5000,-0.6640,-1.1878
// CHECK-NEXT: 0.6000,-0.6640,-1.1878
// CHECK-NEXT: 0.7000,-0.6640,-1.1878
// CHECK-NEXT: 0.8000,-0.6640,-1.1878
// CHECK-NEXT: 0.9000,-0.6640,-1.1878
// CHECK-NEXT: 1.0000,-0.6640,-1.1878

model CycleWithImplicitEquation
    Real x(start = -0.7, fixed = false);
    Real y(start = -0.7, fixed = false);
equation
    x * x + y * y + x + y = 0;
    x * x * y + x = y;
end CycleWithImplicitEquation;
