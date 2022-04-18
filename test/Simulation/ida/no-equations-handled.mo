// RUN: marco --omc-bypass --model=NoEquationsForIDA --end-time=1 --time-step=0.1 --solver=ida --ida-equidistant-time-grid -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s --check-prefix="CHECK-EQUIDISTANT"

// CHECK-EQUIDISTANT: time;x
// CHECK-EQUIDISTANT-NEXT: 0.000000;2.000000
// CHECK-EQUIDISTANT-NEXT: 0.100000;2.000000
// CHECK-EQUIDISTANT-NEXT: 0.200000;2.000000
// CHECK-EQUIDISTANT-NEXT: 0.300000;2.000000
// CHECK-EQUIDISTANT-NEXT: 0.400000;2.000000
// CHECK-EQUIDISTANT-NEXT: 0.500000;2.000000
// CHECK-EQUIDISTANT-NEXT: 0.600000;2.000000
// CHECK-EQUIDISTANT-NEXT: 0.700000;2.000000
// CHECK-EQUIDISTANT-NEXT: 0.800000;2.000000
// CHECK-EQUIDISTANT-NEXT: 0.900000;2.000000
// CHECK-EQUIDISTANT-NEXT: 1.000000;2.000000

// RUN: marco --omc-bypass --model=NoEquationsForIDA --end-time=1 --time-step=0.1 --solver=ida -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s --check-prefix="CHECK-NO-EQUIDISTANT"

// CHECK-NO-EQUIDISTANT: time;x
// CHECK-NO-EQUIDISTANT-NEXT: 0.000000;2.000000
// CHECK-NO-EQUIDISTANT-NEXT: 1.000000;2.000000

model NoEquationsForIDA
	Real x;
equation
	x = 2;
end NoEquationsForIDA;
