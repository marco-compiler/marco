// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.start @r::@x {
// CHECK-DAG:       %[[x_value_1:.*]] = bmodelica.constant #bmodelica.real<1.000000e+00>
// CHECK-DAG:       %[[x_value_2:.*]] = bmodelica.constant #bmodelica.real<2.000000e+00>
// CHECK-DAG:       %[[x_value_3:.*]] = bmodelica.constant #bmodelica.real<3.000000e+00>
// CHECK-NEXT:      %[[x_value:.*]] = bmodelica.array_from_elements %[[x_value_1]], %[[x_value_2]], %[[x_value_3]]
// CHECK-NEXT:      bmodelica.yield %[[x_value]]
// CHECK-NEXT:  }

// CHECK:       bmodelica.start @r::@y {
// CHECK-DAG:       %[[y_value_4:.*]] = bmodelica.constant #bmodelica.real<4.000000e+00>
// CHECK-DAG:       %[[y_value_5:.*]] = bmodelica.constant #bmodelica.real<5.000000e+00>
// CHECK-DAG:       %[[y_value_6:.*]] = bmodelica.constant #bmodelica.real<6.000000e+00>
// CHECK-NEXT:      %[[y_value:.*]] = bmodelica.array_from_elements %[[y_value_4]], %[[y_value_5]], %[[y_value_6]]
// CHECK-NEXT:      bmodelica.yield %[[y_value]]
// CHECK-NEXT:  }

model M
  record R
    Real x;
    Real y;
  end R;

  R[3] r(x(start = {1.0, 2.0, 3.0}), y(start = {4.0, 5.0, 6.0}));
end M;
