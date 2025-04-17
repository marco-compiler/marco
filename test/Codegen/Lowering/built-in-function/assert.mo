// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @assertWarning
// CHECK: bmodelica.assert
// CHECK-SAME: level = 0 : i64
// CHECK-SAME: message = "y was not set to 2"

function assertWarning
    input Integer n1;
    input Integer n2;
    output Real y;
algorithm
    y := n1 + n2;
    assert(y == 5, "y was not set to 2", AssertionLevel.WARNING);
end assertWarning;

// CHECK-LABEL: @assertError
// CHECK: bmodelica.assert
// CHECK-SAME: level = 1 : i64
// CHECK-SAME: message = "y was not set to 2"

function assertError
    input Integer n1;
    input Integer n2;
    output Real y;
algorithm
    y := n1 + n2;
    assert(y == 2, "y was not set to 2", AssertionLevel.ERROR);
end assertError;

// CHECK-LABEL: @assertUnspec
// CHECK: bmodelica.assert
// CHECK-SAME: level = 2 : i64
// CHECK-SAME: message = "y was not set to 2"

function assertUnspec
    input Integer n1;
    input Integer n2;
    output Real y;
algorithm
    y := n1 + n2;
    assert(y == 2, "y was not set to 2");
end assertUnspec;