// RUN: modelica-opt %s --split-input-file --insert-missing-start-values | FileCheck %s

// CHECK-LABEL: @Boolean

bmodelica.model @Boolean {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.bool>

    // CHECK:       bmodelica.start @x
    // CHECK-NEXT:      %[[value:.*]] = bmodelica.constant #bmodelica<bool false> : !bmodelica.bool
    // CHECK-NEXT:      bmodelica.yield %[[value]]
    // CHECK-NEXT:  }
    // CHECK-SAME:  each = false
    // CHECK-SAME:  fixed = false
    // CHECK-SAME:  implicit = true
}

// -----

// CHECK-LABEL: @Integer

bmodelica.model @Integer {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    // CHECK:       bmodelica.start @x
    // CHECK-NEXT:      %[[value:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:      bmodelica.yield %[[value]]
    // CHECK-NEXT:  }
    // CHECK-SAME:  each = false
    // CHECK-SAME:  fixed = false
    // CHECK-SAME:  implicit = true
}

// -----

// CHECK-LABEL: @Real

bmodelica.model @Real {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>

    // CHECK:       bmodelica.start @x
    // CHECK-NEXT:      %[[value:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
    // CHECK-NEXT:      bmodelica.yield %[[value]]
    // CHECK-NEXT:  }
    // CHECK-SAME:  each = false
    // CHECK-SAME:  fixed = false
    // CHECK-SAME:  implicit = true
}

// -----

// CHECK-LABEL: @mlirInteger

bmodelica.model @mlirInteger {
    bmodelica.variable @x : !bmodelica.variable<i64>

    // CHECK:       bmodelica.start @x {
    // CHECK-NEXT:      %[[value:.*]] = bmodelica.constant 0 : i64
    // CHECK-NEXT:      bmodelica.yield %[[value]]
    // CHECK-NEXT:  }
    // CHECK-SAME:  each = false
    // CHECK-SAME:  fixed = false
    // CHECK-SAME:  implicit = true
}

// -----

// CHECK-LABEL: @mlirFloat32

bmodelica.model @mlirFloat32 {
    bmodelica.variable @x : !bmodelica.variable<f32>

    // CHECK:       bmodelica.start @x {
    // CHECK-NEXT:      %[[value:.*]] = bmodelica.constant 0.000000e+00 : f32
    // CHECK-NEXT:      bmodelica.yield %[[value]]
    // CHECK-NEXT:  }
    // CHECK-SAME:  each = false
    // CHECK-SAME:  fixed = false
    // CHECK-SAME:  implicit = true
}

// -----

// CHECK-LABEL: @mlirFloat64

bmodelica.model @mlirFloat64 {
    bmodelica.variable @x : !bmodelica.variable<f64>

    // CHECK:       bmodelica.start @x {
    // CHECK-NEXT:      %[[value:.*]] = bmodelica.constant 0.000000e+00 : f64
    // CHECK-NEXT:      bmodelica.yield %[[value]]
    // CHECK-NEXT:  }
    // CHECK-SAME:  each = false
    // CHECK-SAME:  fixed = false
    // CHECK-SAME:  implicit = true
}
