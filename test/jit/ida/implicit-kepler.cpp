// RUN: marco %s.mo --clever-dae --end-time=10 --time-step=5 --equidistant -o %basename_t.bc
// RUN: clang++ %basename_t.bc %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// 5.0 = x[1:2] - 2.72 * sin(x[1:2])

// CHECK: time,x[1],x[2]
// CHECK-NEXT: 0.000000000000,3.700000000000,3.70000000000
// CHECK-NEXT: 5.000000000000,3.657695461978,3.657695461978
// CHECK-NEXT: 10.000000000000,3.657695461978,3.657695461978
