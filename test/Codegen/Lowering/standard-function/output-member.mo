// RUN: marco %s --omc-bypass --emit-mlir -o %t
// RUN: cat %t | FileCheck %s

// CHECK-LABEL: @outputBooleanScalar
// CHECK: modelica.member_create @y
// CHECK-SAME: !modelica.member<!modelica.bool, output>

function outputBooleanScalar
    output Boolean y;
algorithm
end outputBooleanScalar;


// CHECK-LABEL: @outputIntegerScalar
// CHECK: modelica.member_create @y
// CHECK-SAME: !modelica.member<!modelica.int, output>

function outputIntegerScalar
    output Integer y;
algorithm
end outputIntegerScalar;


// CHECK-LABEL: @outputRealScalar
// CHECK: modelica.member_create @y
// CHECK-SAME: !modelica.member<!modelica.real, output>

function outputRealScalar
    output Real y;
algorithm
end outputRealScalar;


// CHECK-LABEL: @outputBooleanStaticArray
// CHECK: modelica.member_create @y
// CHECK-SAME: !modelica.member<3x2x!modelica.bool, output>

function outputBooleanStaticArray
    output Boolean[3,2] y;
algorithm
end outputBooleanStaticArray;


// CHECK-LABEL: @outputBooleanDynamicArray
// CHECK: modelica.member_create @y
// CHECK-SAME: !modelica.member<?x?x!modelica.bool, output>

function outputBooleanDynamicArray
    output Boolean[:,:] y;
algorithm
end outputBooleanDynamicArray;


// CHECK-LABEL: @outputIntegerStaticArray
// CHECK: modelica.member_create @y
// CHECK-SAME: !modelica.member<3x2x!modelica.int, output>

function outputIntegerStaticArray
    output Integer[3,2] y;
algorithm
end outputIntegerStaticArray;


// CHECK-LABEL: @outputIntegerDynamicArray
// CHECK: modelica.member_create @y
// CHECK-SAME: !modelica.member<?x?x!modelica.int, output>

function outputIntegerDynamicArray
    output Integer[:,:] y;
algorithm
end outputIntegerDynamicArray;


// CHECK-LABEL: @outputRealStaticArray
// CHECK: modelica.member_create @y
// CHECK-SAME: !modelica.member<3x2x!modelica.real, output>

function outputRealStaticArray
    output Real[3,2] y;
algorithm
end outputRealStaticArray;


// CHECK-LABEL: @outputRealDynamicArray
// CHECK: modelica.member_create @y
// CHECK-SAME: !modelica.member<?x?x!modelica.real, output>

function outputRealDynamicArray
    output Real[:,:] y;
algorithm
end outputRealDynamicArray;


// CHECK-LABEL: @sizeDependingOnIntegerInput
// CHECK: %[[x:.*]] = modelica.member_create @n
// CHECK-SAME: !modelica.member<!modelica.int, input>
// CHECK: %[[x_value:.*]] = modelica.member_load %[[x]]
// CHECK: %[[size:.*]] = modelica.cast %[[x_value]] : !modelica.int -> index
// CHECK: %[[y:.*]] = modelica.member_create @y
// CHECK-SAME: %[[size]]
// CHECK-SAME: !modelica.member<?x!modelica.real, output>

function sizeDependingOnIntegerInput
    input Integer n;
    output Real[n] y;
algorithm
end sizeDependingOnIntegerInput;


// CHECK-LABEL: @defaultValue
// CHECK: %[[x:.*]] = modelica.member_create @x
// CHECK-SAME: !modelica.member<!modelica.int, output>
// CHECK: %[[value:.*]] = modelica.constant #modelica.int<10>
// CHECK: modelica.member_store %[[x]], %[[value]]

function defaultValue
    output Integer x = 10;
algorithm
end defaultValue;
