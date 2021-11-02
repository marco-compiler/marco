// RUN: marco %s.mo --clever-dae --end-time=10 --rel-tol=1e-10 --abs-tol=1e-10 -o %basename_t.bc
// RUN: clang++ %basename_t.bc %runtime_lib -o %t
// RUN: %t | FileCheck %s

// x[1:6] = i + sin(time * 100)
// der(y[1:3]) = x[3:6]
// y[1:3] = i * time - cos(100 * time) / 100

// CHECK: time,x[1],x[2],x[3],x[4],x[5],x[6],y[1],y[2],y[3]

// CHECK-NEXT: 0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,
// CHECK-SAME: 0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000

// CHECK: 10.000000000000

// CHECK-SAME: 1.82687954
// CHECK-SAME: 2.82687954
// CHECK-SAME: 3.82687954
// CHECK-SAME: 4.82687954
// CHECK-SAME: 5.82687954
// CHECK-SAME: 6.82687954

// CHECK-SAME: 40.0043762
// CHECK-SAME: 50.0043762
// CHECK-SAME: 60.0043762
