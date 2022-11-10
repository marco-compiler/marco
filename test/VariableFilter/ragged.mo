// RUN: marco --omc-bypass --model=Ragged  --variable-filter="time;s[2:2,1:1];x"  --solver=ida -o %basename_t %s
// RUN: ./%basename_t --time-step=0.1 --end-time=1 --precision=6 | FileCheck %s

// CHECK: "time","s[2][1]","x"
// CHECK-NEXT: 0.000000,1.000000,0.000000
// CHECK-NEXT: 0.100000,1.100000,0.005000
// CHECK-NEXT: 0.200000,1.200000,0.020000
// CHECK-NEXT: 0.300000,1.300000,0.045000
// CHECK-NEXT: 0.400000,1.400000,0.080000
// CHECK-NEXT: 0.500000,1.500000,0.125000
// CHECK-NEXT: 0.600000,1.600000,0.180000
// CHECK-NEXT: 0.700000,1.700000,0.245000
// CHECK-NEXT: 0.800000,1.800000,0.320000
// CHECK-NEXT: 0.900000,1.900000,0.405000
// CHECK-NEXT: 1.000000,2.000000,0.500000

model Ragged
  Real[2,{1,2}] s;
	Real x(start = 0, fixed = true);
initial equation
equation
  s[1,1] = time;
  s[2,1] = s[1,1]+1.0;
  s[2,2] = s[2,1]+2.0;
	der(x) = s[2,2]-3.0;
end Ragged;


