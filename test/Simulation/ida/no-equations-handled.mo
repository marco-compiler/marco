// RUN: marco %s --omc-bypass --model=NoEquationsForIDA --solver=ida -o %basename_t %link_flags -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s --check-prefix="CHECK-EQUIDISTANT"

// CHECK-EQUIDISTANT: "time","x"
// CHECK-EQUIDISTANT: 0.0000,2.0000
// CHECK-EQUIDISTANT: 1.0000,2.0000

// RUN: marco %s --omc-bypass --model=NoEquationsForIDA --solver=ida -o %basename_t -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --precision=6 | FileCheck %s --check-prefix="CHECK-NO-EQUIDISTANT"

// CHECK-NO-EQUIDISTANT: "time","x"
// CHECK-NO-EQUIDISTANT-NEXT: 0.000000,2.000000
// CHECK-NO-EQUIDISTANT-NEXT: 1.000000,2.000000

model NoEquationsForIDA
	Real x;
equation
	x = 2;
end NoEquationsForIDA;
