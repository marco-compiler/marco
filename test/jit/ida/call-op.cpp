// RUN: marco %s.mo --clever-dae --end-time=10 --time-step=5 --rel-tol=1e-10 --abs-tol=1e-10 --equidistant -o %basename_t.bc
// RUN: clang++ %basename_t.bc %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// x[1] + x[2] = 2;
// x[1] - x[2] = 4;
// x[3] + x[4] = 1;
// x[3] - x[4] = -1;
// x[5] = x[4] + x[1];

// CHECK: time,x[1],y[1]
// CHECK-NEXT: 0.000000000000,0.000000000000,0.000000000000
// CHECK-NEXT: 5.000000000000,10.000000000000,25.91953
// CHECK-NEXT: 10.000000000000,20.000000000000,100.29595
