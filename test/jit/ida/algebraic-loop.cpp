// RUN: marco %s.mo --clever-dae --end-time=10 --time-step=5 --equidistant -o %basename_t.bc
// RUN: clang++ %basename_t.bc %runtime_lib -o %t
// RUN: %t | FileCheck %s

// x[1] + x[2] = 2;
// x[1] - x[2] = 4;
// x[3] + x[4] = 1;
// x[3] - x[4] = -1;
// x[5] = x[4] + x[1];

// CHECK: time,x[1],x[2],x[3],x[4],x[5]
// CHECK-NEXT: 0.000000000000,0,0,0,0,0
// CHECK-NEXT: 5.000000000000,3,-1,0,1,4
// CHECK-NEXT: 10.000000000000,3,-1,0,1,4
