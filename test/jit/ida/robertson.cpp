// RUN: marco %s.mo --clever-dae --end-time=400 --rel-tol=1e-10 --abs-tol=1e-10 -o %basename_t.bc
// RUN: clang++ %basename_t.bc %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// der(y1) = -0.04 * y1 + 1e4 * y2 * y3;
// der(y2) = +0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2 * y2;
// 0 = y1 + y2 + y3 - 1;

// CHECK: time,y1[1],y2[1],y3[1]
// CHECK-NEXT: 0.000000000000,1.000000000000,0.000000000000,0.000000000000

// CHECK: 400.000000000000
// CHECK-SAME: 0.4505186
// CHECK-SAME: 0.0000032
// CHECK-SAME: 0.5494781
