// RUN: marco %s.mo --clever-dae --end-time=10 --time-step=5 --equidistant -o %basename_t.bc
// RUN: clang++ %basename_t.bc %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// x + y = 1;
// x*x*x + x = y;

// CHECK: time,x[1],y[1]
// CHECK-NEXT: 0.000000000000,0.000000000000,0.000000000000

// CHECK: 5.000000000000
// CHECK-SAME: 0.453397651
// CHECK-SAME: 0.546602348

// CHECK: 10.000000000000
// CHECK-SAME: 0.453397651
// CHECK-SAME: 0.546602348
