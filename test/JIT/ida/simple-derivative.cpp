// RUN: marco %s.mo --clever-dae --end-time=10 --time-step=5 --equidistant -o %basename_t.bc
// RUN: clang++ %basename_t.bc %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// x = 2 * time

// CHECK: time,x[1]
// CHECK-NEXT: 0.000000000000,0.000000000000
// CHECK-NEXT: 5.000000000000,10.000000000000
// CHECK-NEXT: 10.000000000000,20.000000000000
