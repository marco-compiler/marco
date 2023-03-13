// RUN: marco %s --omc-bypass --emit-mlir -o %t
// RUN: cat %t | FileCheck %s

// CHECK-LABEL: @outputBooleanScalar
// CHECK: modelica.variable @y : !modelica.variable<!modelica.bool, output>

function outputBooleanScalar
    output Boolean y;
algorithm
end outputBooleanScalar;


// CHECK-LABEL: @outputIntegerScalar
// CHECK: modelica.variable @y : !modelica.variable<!modelica.int, output>

function outputIntegerScalar
    output Integer y;
algorithm
end outputIntegerScalar;


// CHECK-LABEL: @outputRealScalar
// CHECK: modelica.variable @y : !modelica.variable<!modelica.real, output>

function outputRealScalar
    output Real y;
algorithm
end outputRealScalar;


// CHECK-LABEL: @outputBooleanStaticArray
// CHECK: modelica.variable @y : !modelica.variable<3x2x!modelica.bool, output>

function outputBooleanStaticArray
    output Boolean[3,2] y;
algorithm
end outputBooleanStaticArray;


// CHECK-LABEL: @outputBooleanDynamicArray
// CHECK: modelica.variable @y : !modelica.variable<?x?x!modelica.bool, output>

function outputBooleanDynamicArray
    output Boolean[:,:] y;
algorithm
end outputBooleanDynamicArray;


// CHECK-LABEL: @outputIntegerStaticArray
// CHECK: modelica.variable @y : !modelica.variable<3x2x!modelica.int, output>

function outputIntegerStaticArray
    output Integer[3,2] y;
algorithm
end outputIntegerStaticArray;


// CHECK-LABEL: @outputIntegerDynamicArray
// CHECK: modelica.variable @y : !modelica.variable<?x?x!modelica.int, output>

function outputIntegerDynamicArray
    output Integer[:,:] y;
algorithm
end outputIntegerDynamicArray;


// CHECK-LABEL: @outputRealStaticArray
// CHECK: modelica.variable @y : !modelica.variable<3x2x!modelica.real, output>

function outputRealStaticArray
    output Real[3,2] y;
algorithm
end outputRealStaticArray;


// CHECK-LABEL: @outputRealDynamicArray
// CHECK: modelica.variable @y : !modelica.variable<?x?x!modelica.real, output>

function outputRealDynamicArray
    output Real[:,:] y;
algorithm
end outputRealDynamicArray;


// CHECK-LABEL: @sizeDependingOnIntegerInput
// CHECK: modelica.variable @n : !modelica.variable<!modelica.int, input>
// CHECK:       modelica.variable @y : !modelica.variable<?x!modelica.real, output> [fixed] {
// CHECK-NEXT:      %[[n:.*]] = modelica.variable_get @n
// CHECK-NEXT:      %[[size:.*]] = modelica.cast %[[n]] : !modelica.int -> index
// CHECK-NEXT:      modelica.yield %[[size]]
// CHECK-NEXT:  }

function sizeDependingOnIntegerInput
    input Integer n;
    output Real[n] y;
algorithm
end sizeDependingOnIntegerInput;


// CHECK-LABEL: @defaultValue
// CHECK:       modelica.variable @x : !modelica.variable<!modelica.int, output>
// CHECK-NEXT:  modelica.default @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<10>
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  }

function defaultValue
    output Integer x = 10;
algorithm
end defaultValue;
