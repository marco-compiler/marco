// RUN: modelica-opt %s --split-input-file --function-default-values-conversion | FileCheck %s

// Scalar variable.

// CHECK:       bmodelica.function @foo {
// CHECK:           bmodelica.algorithm {
// CHECK-NEXT:          %[[default:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-NEXT:          bmodelica.variable_set @x, %[[default]]
// CHECK-NEXT:          %[[non_default:.*]] = bmodelica.constant #bmodelica.int<1>
// CHECK-NEXT:          bmodelica.variable_set @x, %[[non_default]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, output>

    bmodelica.default @x {
        %0 = bmodelica.constant #bmodelica.int<0>
        bmodelica.yield %0 : !bmodelica.int
    }

    bmodelica.algorithm {
        %0 = bmodelica.constant #bmodelica.int<1>
        bmodelica.variable_set @x, %0 : !bmodelica.int
    }
}

// -----

// Array variable.

// CHECK:       bmodelica.function @foo {
// CHECK:           bmodelica.algorithm {
// CHECK-NEXT:          %[[default:.*]] = bmodelica.constant #bmodelica.int_array<[0, 0, 0]> : !bmodelica.array<3x!bmodelica.int>
// CHECK-NEXT:          bmodelica.variable_set @x, %[[default]]
// CHECK-NEXT:          %[[non_default:.*]] = bmodelica.constant #bmodelica.int_array<[1, 1, 1]> : !bmodelica.array<3x!bmodelica.int>
// CHECK-NEXT:          bmodelica.variable_set @x, %[[non_default]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, output>

    bmodelica.default @x {
        %0 = bmodelica.constant #bmodelica.int_array<[0, 0, 0]> : !bmodelica.array<3x!bmodelica.int>
        bmodelica.yield %0 : !bmodelica.array<3x!bmodelica.int>
    }

    bmodelica.algorithm {
        %0 = bmodelica.constant #bmodelica.int_array<[1, 1, 1]> : !bmodelica.array<3x!bmodelica.int>
        bmodelica.variable_set @x, %0 : !bmodelica.array<3x!bmodelica.int>
    }
}

// -----

// Missing algorithm.

// CHECK:       bmodelica.function @foo {
// CHECK:           bmodelica.algorithm {
// CHECK-NEXT:          %[[default:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-NEXT:          bmodelica.variable_set @x, %[[default]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    bmodelica.default @x {
        %0 = bmodelica.constant #bmodelica.int<0>
        bmodelica.yield %0 : !bmodelica.int
    }
}
