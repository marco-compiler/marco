// RUN: marco %s.mo --clever-dae --end-time=10 --time-step=5 --equidistant -o %basename_t.bc
// RUN: clang++ %basename_t.bc %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK: time,w[1],z[1]
// CHECK-NEXT: 0.000000000000,0.000000000000,0.000000000000
// CHECK-NEXT: 5.000000000000,799.800000000000,266.600000000000
// CHECK-NEXT: 10.000000000000,1599.600000000000,533.200000000000
