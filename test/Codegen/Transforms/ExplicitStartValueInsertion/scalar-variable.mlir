// RUN: modelica-opt %s --split-input-file --insert-missing-start-values | FileCheck %s

// Uninitialized scalar variable.

// Modelica Boolean type.

// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.bool<false> : !modelica.bool
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = false}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.bool>
}

// -----

// Modelica Integer type.

// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<0> : !modelica.int
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = false}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
}

// -----

// Modelica Real type.

// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = false}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.real>
}

// -----

// Built-in integer type.

// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant 0 : i64
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = false}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<i64>
}

// -----

// Built-in f32 type.

// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant 0.000000e+00 : f32
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = false}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<f32>
}

// -----

// Built-in f64 type.

// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant 0.000000e+00 : f64
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = false}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<f64>
}
