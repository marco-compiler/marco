// RUN: marco %s --omc-bypass --model=CycleWithImplicitEquation --solver=ida -o %basename_t %link_flags -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x","y"
// CHECK: 0.0000,-0.6640,-1.1878
// CHECK: 1.0000,-0.6640,-1.1878

model CycleWithImplicitEquation
    Real x(start = -0.7, fixed = false);
    Real y(start = -0.7, fixed = false);
equation
    x * x + y * y + x + y = 0;
    x * x * y + x = y;
end CycleWithImplicitEquation;
