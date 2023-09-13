// RUN: marco --omc-bypass --model=NoEquationsForIDA --solver=ida -o %basename_t %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s --check-prefix="CHECK-EQUIDISTANT"

// CHECK-EQUIDISTANT: "time","x"
// CHECK-EQUIDISTANT-NEXT: 0.0000,2.0000
// CHECK-EQUIDISTANT-NEXT: 0.1000,2.0000
// CHECK-EQUIDISTANT-NEXT: 0.2000,2.0000
// CHECK-EQUIDISTANT-NEXT: 0.3000,2.0000
// CHECK-EQUIDISTANT-NEXT: 0.4000,2.0000
// CHECK-EQUIDISTANT-NEXT: 0.5000,2.0000
// CHECK-EQUIDISTANT-NEXT: 0.6000,2.0000
// CHECK-EQUIDISTANT-NEXT: 0.7000,2.0000
// CHECK-EQUIDISTANT-NEXT: 0.8000,2.0000
// CHECK-EQUIDISTANT-NEXT: 0.9000,2.0000
// CHECK-EQUIDISTANT-NEXT: 1.0000,2.0000

// RUN: marco --omc-bypass --model=NoEquationsForIDA --solver=ida -o %basename_t %s
// RUN: ./%basename_t --end-time=1 --precision=6 | FileCheck %s --check-prefix="CHECK-NO-EQUIDISTANT"

// CHECK-NO-EQUIDISTANT: "time","x"
// CHECK-NO-EQUIDISTANT-NEXT: 0.000000,2.000000
// CHECK-NO-EQUIDISTANT-NEXT: 1.000000,2.000000

model NoEquationsForIDA
	Real x;
equation
	x = 2;
end NoEquationsForIDA;
