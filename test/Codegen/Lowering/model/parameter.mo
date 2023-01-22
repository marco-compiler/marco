// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: modelica.member_create @n : !modelica.member<!modelica.int, parameter>

model Test
    parameter Integer n;
end Test;
