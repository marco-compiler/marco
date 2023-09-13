// RUN: marco --omc-bypass --model=Robertson --solver=ida -o %basename_t %s
// RUN: ./%basename_t --end-time=400 --time-step=1 --precision=4 | FileCheck %s

// CHECK: "time","y1","y2","y3"
// CHECK-NEXT: 0.0000,1.0000,0.0000,0.0000
// CHECK: 300.0000,0.4862,0.0000,0.5138
// CHECK: 400.0000,0.4505,0.0000,0.5495

model Robertson
	Real y1(start = 1.0, fixed = true);
	Real y2(start = 0.0, fixed = true);
	Real y3(start = 0.0, fixed = false);
equation
	der(y1) = -0.04 * y1 + 1e4 * y2 * y3;
	der(y2) = +0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2 * y2;
	0 = y1 + y2 + y3 - 1;
end Robertson;
