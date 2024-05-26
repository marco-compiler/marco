// RUN: modelica-opt %s --split-input-file --insert-missing-start-values | FileCheck %s

// Uninitialized scalar variable.

// Modelica Boolean type.

// CHECK:       bmodelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = bmodelica.constant #bmodelica<bool false> : !bmodelica.bool
// CHECK-NEXT:      bmodelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = false, implicit = true}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.bool>
}

// -----

// Modelica Integer type.

// CHECK:       bmodelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
// CHECK-NEXT:      bmodelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = false, implicit = true}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
}

// -----

// Modelica Real type.

// CHECK:       bmodelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
// CHECK-NEXT:      bmodelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = false, implicit = true}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
}

// -----

// Built-in integer type.

// CHECK:       bmodelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = bmodelica.constant 0 : i64
// CHECK-NEXT:      bmodelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = false, implicit = true}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<i64>
}

// -----

// Built-in f32 type.

// CHECK:       bmodelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = bmodelica.constant 0.000000e+00 : f32
// CHECK-NEXT:      bmodelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = false, implicit = true}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<f32>
}

// -----

// Built-in f64 type.

// CHECK:       bmodelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = bmodelica.constant 0.000000e+00 : f64
// CHECK-NEXT:      bmodelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = false, implicit = true}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<f64>
}
