// RUN: marco %s --model=Test --solver=euler-forward -o %basename_t %link_flags -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=6 | FileCheck %s

// XFAIL: *
// Record constructors are not supported yet by the native frontend.

// CHECK: "time","z1.im","z1.re","z2.im","z2.re"
// CHECK: 0.000000,2.000000,1.000000,4.000000,3.000000
// CHECK: 1.000000,2.000000,1.000000,4.000000,3.000000

record Complex
    Real re;
    Real im;
end Complex;

model Test
    parameter Complex z1 = Complex(1.0, 2.0);
    Complex z2;
equation
    z2 = Complex(3.0, 4.0);
end Test;
