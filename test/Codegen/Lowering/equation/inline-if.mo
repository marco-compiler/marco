// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK: bmodelica.equation
// CHECK-DAG: %[[condition:.*]] = bmodelica.gt
// CHECK-DAG: %[[trueValue:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK-DAG: %[[falseValue:.*]] = bmodelica.constant #bmodelica<int 0>
// CHECK: %[[select:.*]] = bmodelica.select (%[[condition]] : !bmodelica.bool), (%[[trueValue]] : !bmodelica.int), (%[[falseValue]] : !bmodelica.int) -> !bmodelica.int
// CHECK: %[[rhs:.*]] = bmodelica.equation_side %[[select]]
// CHECK: bmodelica.equation_sides %{{.*}}, %[[rhs]]

model M1
	Real x;
equation
   x = if time > 0.5 then 1 else 0;
end M1;
