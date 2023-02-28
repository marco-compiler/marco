// RUN: marco %s --omc-bypass --emit-mlir -o %t
// RUN: cat %t | FileCheck %s

// CHECK-LABEL: @inputVariablesRelativeOrder
// CHECK: modelica.member_create @x
// CHECK: modelica.member_create @y

function inputVariablesRelativeOrder
    input Integer x;
    input Integer y;
algorithm
end inputVariablesRelativeOrder;


// CHECK-LABEL: @outputVariablesRelativeOrder
// CHECK: modelica.member_create @x
// CHECK: modelica.member_create @y

function outputVariablesRelativeOrder
    output Integer x;
    output Integer y;
algorithm
end outputVariablesRelativeOrder;


// CHECK-LABEL: @outputVariableDependingOnInputVariable
// CHECK: modelica.member_create @x
// CHECK: modelica.member_create @y

function outputVariableDependingOnInputVariable
    input Integer[:] x;
    output Integer[size(x, 1)] y;
algorithm
end outputVariableDependingOnInputVariable;


// CHECK-LABEL: @outputVariableDependingOnFollowingInputVariable
// CHECK: modelica.member_create @x
// CHECK: modelica.member_create @y

function outputVariableDependingOnFollowingInputVariable
    output Integer[size(x, 1)] y;
    input Integer[:] x;
algorithm
end outputVariableDependingOnFollowingInputVariable;
