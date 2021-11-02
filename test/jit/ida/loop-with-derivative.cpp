// RUN: marco %s.mo --clever-dae --end-time=10 --time-step=5 --equidistant -o %basename_t.bc
// RUN: clang++ %basename_t.bc %runtime_lib -o %t
// RUN: %t | FileCheck %s

// x = 3 * time
// y = 1

// CHECK: time,x[1],y[1]
// CHECK-NEXT: 0.000000000000,0.000000000000,0.000000000000
// CHECK-NEXT: 5.000000000000,15.000000000000,1.000000000000
// CHECK-NEXT: 10.000000000000,30.000000000000,1.000000000000
