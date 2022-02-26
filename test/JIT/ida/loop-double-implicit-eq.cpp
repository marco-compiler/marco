// RUN: marco %s.mo --clever-dae --end-time=10 --time-step=5 --equidistant -o %basename_t.bc
// RUN: clang++ %basename_t.bc %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// x * x + y * y + x + y = 0;
// x * x * y + x = y;

// CHECK: time,x[1],y[1]
// CHECK-NEXT: 0.000000000000,-2.000000000000,-2.000000000000

// CHECK: 5.000000000000
// CHECK-SAME: -0.6640
// CHECK-SAME: -1.1878

// CHECK: 10.000000000000
// CHECK-SAME: -0.6640
// CHECK-SAME: -1.1878
