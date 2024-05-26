// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @outputBooleanScalar
// CHECK: bmodelica.variable @y : !bmodelica.variable<!bmodelica.bool, output>

function outputBooleanScalar
    output Boolean y;
algorithm
end outputBooleanScalar;


// CHECK-LABEL: @outputIntegerScalar
// CHECK: bmodelica.variable @y : !bmodelica.variable<!bmodelica.int, output>

function outputIntegerScalar
    output Integer y;
algorithm
end outputIntegerScalar;


// CHECK-LABEL: @outputRealScalar
// CHECK: bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

function outputRealScalar
    output Real y;
algorithm
end outputRealScalar;


// CHECK-LABEL: @outputBooleanStaticArray
// CHECK: bmodelica.variable @y : !bmodelica.variable<3x2x!bmodelica.bool, output>

function outputBooleanStaticArray
    output Boolean[3,2] y;
algorithm
end outputBooleanStaticArray;


// CHECK-LABEL: @outputBooleanDynamicArray
// CHECK: bmodelica.variable @y : !bmodelica.variable<?x?x!bmodelica.bool, output>

function outputBooleanDynamicArray
    output Boolean[:,:] y;
algorithm
end outputBooleanDynamicArray;


// CHECK-LABEL: @outputIntegerStaticArray
// CHECK: bmodelica.variable @y : !bmodelica.variable<3x2x!bmodelica.int, output>

function outputIntegerStaticArray
    output Integer[3,2] y;
algorithm
end outputIntegerStaticArray;


// CHECK-LABEL: @outputIntegerDynamicArray
// CHECK: bmodelica.variable @y : !bmodelica.variable<?x?x!bmodelica.int, output>

function outputIntegerDynamicArray
    output Integer[:,:] y;
algorithm
end outputIntegerDynamicArray;


// CHECK-LABEL: @outputRealStaticArray
// CHECK: bmodelica.variable @y : !bmodelica.variable<3x2x!bmodelica.real, output>

function outputRealStaticArray
    output Real[3,2] y;
algorithm
end outputRealStaticArray;


// CHECK-LABEL: @outputRealDynamicArray
// CHECK: bmodelica.variable @y : !bmodelica.variable<?x?x!bmodelica.real, output>

function outputRealDynamicArray
    output Real[:,:] y;
algorithm
end outputRealDynamicArray;


// CHECK-LABEL: @sizeDependingOnIntegerInput
// CHECK: bmodelica.variable @n : !bmodelica.variable<!bmodelica.int, input>
// CHECK:       bmodelica.variable @y : !bmodelica.variable<?x!bmodelica.real, output> [fixed] {
// CHECK-NEXT:      %[[n:.*]] = bmodelica.variable_get @n
// CHECK-NEXT:      %[[size:.*]] = bmodelica.cast %[[n]] : !bmodelica.int -> index
// CHECK-NEXT:      bmodelica.yield %[[size]]
// CHECK-NEXT:  }

function sizeDependingOnIntegerInput
    input Integer n;
    output Real[n] y;
algorithm
end sizeDependingOnIntegerInput;


// CHECK-LABEL: @defaultValue
// CHECK:       bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, output>
// CHECK-NEXT:  bmodelica.default @x {
// CHECK-NEXT:      %[[value:.*]] = bmodelica.constant #bmodelica<int 10>
// CHECK-NEXT:      bmodelica.yield %[[value]]
// CHECK-NEXT:  }

function defaultValue
    output Integer x = 10;
algorithm
end defaultValue;
