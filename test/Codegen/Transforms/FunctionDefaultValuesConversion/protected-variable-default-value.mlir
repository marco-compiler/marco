// RUN: modelica-opt %s --split-input-file --function-default-values-conversion | FileCheck %s

// Scalar variable.

// CHECK:       modelica.function @foo {
// CHECK:           modelica.algorithm {
// CHECK-NEXT:          %[[default:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:          modelica.variable_set @x, %[[default]]
// CHECK-NEXT:          %[[non_default:.*]] = modelica.constant #modelica.int<1>
// CHECK-NEXT:          modelica.variable_set @x, %[[non_default]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

modelica.function @foo {
    modelica.variable @x : !modelica.variable<!modelica.int>

    modelica.default @x {
        %0 = modelica.constant #modelica.int<0>
        modelica.yield %0 : !modelica.int
    }

    modelica.algorithm {
        %0 = modelica.constant #modelica.int<1>
        modelica.variable_set @x, %0 : !modelica.int
    }
}

// -----

// Array variable.

// CHECK:       modelica.function @foo {
// CHECK:           modelica.algorithm {
// CHECK-NEXT:          %[[default:.*]] = modelica.constant #modelica.int_array<[0, 0, 0]> : !modelica.array<3x!modelica.int>
// CHECK-NEXT:          modelica.variable_set @x, %[[default]]
// CHECK-NEXT:          %[[non_default:.*]] = modelica.constant #modelica.int_array<[1, 1, 1]> : !modelica.array<3x!modelica.int>
// CHECK-NEXT:          modelica.variable_set @x, %[[non_default]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

modelica.function @foo {
    modelica.variable @x : !modelica.variable<3x!modelica.int>

    modelica.default @x {
        %0 = modelica.constant #modelica.int_array<[0, 0, 0]> : !modelica.array<3x!modelica.int>
        modelica.yield %0 : !modelica.array<3x!modelica.int>
    }

    modelica.algorithm {
        %0 = modelica.constant #modelica.int_array<[1, 1, 1]> : !modelica.array<3x!modelica.int>
        modelica.variable_set @x, %0 : !modelica.array<3x!modelica.int>
    }
}

// -----

// Missing algorithm.

// CHECK:       modelica.function @foo {
// CHECK:           modelica.algorithm {
// CHECK-NEXT:          %[[default:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:          modelica.variable_set @x, %[[default]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

modelica.function @foo {
    modelica.variable @x : !modelica.variable<!modelica.int>

    modelica.default @x {
        %0 = modelica.constant #modelica.int<0>
        modelica.yield %0 : !modelica.int
    }
}
