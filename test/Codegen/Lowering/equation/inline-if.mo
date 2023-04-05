// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK: modelica.equation
// CHECK-DAG: %[[condition:.*]] = modelica.gt
// CHECK-DAG: %[[trueValue:.*]] = modelica.constant #modelica.int<1>
// CHECK-DAG: %[[falseValue:.*]] = modelica.constant #modelica.int<0>
// CHECK: %[[select:.*]] = modelica.select (%[[condition]] : !modelica.bool), (%[[trueValue]] : !modelica.int), (%[[falseValue]] : !modelica.int) -> !modelica.int
// CHECK: %[[rhs:.*]] = modelica.equation_side %[[select]]
// CHECK: modelica.equation_sides %{{.*}}, %[[rhs]]

model M1
	Real x;
equation
   x = if time > 0.5 then 1 else 0;
end M1;
