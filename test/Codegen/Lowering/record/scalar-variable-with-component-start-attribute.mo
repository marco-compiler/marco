// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.start @r::@x {
// CHECK-NEXT:      %[[x_value:.*]] = bmodelica.constant #bmodelica.real<1.000000e+00>
// CHECK-NEXT:      bmodelica.yield %[[x_value]]
// CHECK-NEXT:  }

// CHECK:       bmodelica.start @r::@y {
// CHECK-NEXT:      %[[y_value:.*]] = bmodelica.constant #bmodelica.real<2.000000e+00>
// CHECK-NEXT:      bmodelica.yield %[[y_value]]
// CHECK-NEXT:  }

model M
  record R
    Real x;
    Real y;
  end R;

  R r(x(start = 1.0), y(start = 2.0));
end M;
