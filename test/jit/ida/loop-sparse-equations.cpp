// RUN: marco %s.mo --clever-dae --end-time=10 --time-step=5 --equidistant -o %basename_t.bc
// RUN: clang++ %basename_t.bc %runtime_lib -o %t
// RUN: %t | FileCheck %s

// x[1] + x[2] = 3;
// x[1] + x[3] = 2;
// x[2] + x[3] = 1;

// CHECK: time,x[1],x[2],x[3]
// CHECK-NEXT: 0.000000000000,0.000000000000,0.000000000000,0.000000000000
// CHECK-NEXT: 5.000000000000,2.000000000000,1.000000000000,0.000000000000
// CHECK-NEXT: 10.000000000000,2.000000000000,1.000000000000,0.000000000000
