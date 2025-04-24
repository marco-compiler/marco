// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @warning
// CHECK: bmodelica.assert
// CHECK-SAME: level = 0
// CHECK-SAME: message = "message"

function warning
algorithm
    assert(false, "message", AssertionLevel.warning);
end warning;

// CHECK-LABEL: @error
// CHECK: bmodelica.assert
// CHECK-SAME: level = 1
// CHECK-SAME: message = "message"

function error
algorithm
    assert(false, "message", AssertionLevel.error);
end error;

// CHECK-LABEL: @integerCondition
// CHECK: bmodelica.assert
// CHECK: %[[cast:.*]] = bmodelica.cast
// CHECK-SAME: !bmodelica.int -> !bmodelica.bool
// CHECK: bmodelica.yield %[[cast]]

function integerCondition
algorithm
    assert(0, "message", AssertionLevel.error);
end integerCondition;

// CHECK-LABEL: @realCondition
// CHECK: bmodelica.assert
// CHECK: %[[cast:.*]] = bmodelica.cast
// CHECK-SAME: !bmodelica.real -> !bmodelica.bool
// CHECK: bmodelica.yield %[[cast]]

function realCondition
algorithm
    assert(0.0, "message", AssertionLevel.error);
end realCondition;
