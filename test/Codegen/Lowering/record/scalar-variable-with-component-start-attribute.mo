// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       modelica.start @r::@x {
// CHECK-NEXT:      %[[x_value:.*]] = modelica.constant #modelica.real<1.000000e+00>
// CHECK-NEXT:      modelica.yield %[[x_value]]
// CHECK-NEXT:  }

// CHECK:       modelica.start @r::@y {
// CHECK-NEXT:      %[[y_value:.*]] = modelica.constant #modelica.real<2.000000e+00>
// CHECK-NEXT:      modelica.yield %[[y_value]]
// CHECK-NEXT:  }

model M
  record R
    Real x;
    Real y;
  end R;

  R r(x(start = 1.0), y(start = 2.0));
end M;
