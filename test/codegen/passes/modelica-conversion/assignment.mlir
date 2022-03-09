// RUN: modelica-opt %s --convert-modelica | FileCheck %s

// CHECK-LABEL: @castIntegerToReal
// CHECK-SAME: %arg0: !modelica.int
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.alloca : !modelica.array<stack, !modelica.real>
// CHECK: %[[ARG0_CASTED:[a-zA-Z0-9]*]] = modelica.cast %arg0 : !modelica.int -> !modelica.real
// CHECK: modelica.store %[[Y]][], %[[ARG0_CASTED]] : !modelica.array<stack, !modelica.real>

func @castIntegerToReal(%arg0: !modelica.int) -> () {
    %y = modelica.alloca : !modelica.array<stack, !modelica.real>
    modelica.assignment %y, %arg0 : !modelica.array<stack, !modelica.real>, !modelica.int
    return
}

// CHECK-LABEL: @castRealToInteger
// CHECK-SAME: %arg0: !modelica.real
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.alloca : !modelica.array<stack, !modelica.int>
// CHECK: %[[ARG0_CASTED:[a-zA-Z0-9]*]] = modelica.cast %arg0 : !modelica.real -> !modelica.int
// CHECK: modelica.store %[[Y]][], %[[ARG0_CASTED]] : !modelica.array<stack, !modelica.int>

func @castRealToInteger(%arg0: !modelica.real) -> () {
    %y = modelica.alloca : !modelica.array<stack, !modelica.int>
    modelica.assignment %y, %arg0 : !modelica.array<stack, !modelica.int>, !modelica.real
    return
}

