// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @outputBooleanScalar
// CHECK: modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<!modelica.bool, output>

function outputBooleanScalar
    output Boolean y;

algorithm
end outputBooleanScalar;


// CHECK-LABEL: @outputIntegerScalar
// CHECK: modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<!modelica.int, output>

function outputIntegerScalar
    output Integer y;

algorithm
end outputIntegerScalar;


// CHECK-LABEL: @outputRealScalar
// CHECK: modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<!modelica.real, output>

function outputRealScalar
    output Real y;

algorithm
end outputRealScalar;


// CHECK-LABEL: @outputBooleanStaticArray
// CHECK: modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<3x2x!modelica.bool, output>

function outputBooleanStaticArray
    output Boolean[3,2] y;

algorithm
end outputBooleanStaticArray;


// CHECK-LABEL: @outputBooleanDynamicArray
// CHECK: modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<?x?x!modelica.bool, output>

function outputBooleanDynamicArray
    output Boolean[:,:] y;

algorithm
end outputBooleanDynamicArray;


// CHECK-LABEL: @outputIntegerStaticArray
// CHECK: modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<3x2x!modelica.int, output>

function outputIntegerStaticArray
    output Integer[3,2] y;

algorithm
end outputIntegerStaticArray;


// CHECK-LABEL: @outputIntegerDynamicArray
// CHECK: modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<?x?x!modelica.int, output>

function outputIntegerDynamicArray
    output Integer[:,:] y;

algorithm
end outputIntegerDynamicArray;


// CHECK-LABEL: @outputRealStaticArray
// CHECK: modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<3x2x!modelica.real, output>

function outputRealStaticArray
    output Real[3,2] y;

algorithm
end outputRealStaticArray;


// CHECK-LABEL: @outputRealDynamicArray
// CHECK: modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<?x?x!modelica.real, output>

function outputRealDynamicArray
    output Real[:,:] y;

algorithm
end outputRealDynamicArray;
