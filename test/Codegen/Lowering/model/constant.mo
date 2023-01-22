// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: modelica.member_create @n : !modelica.member<!modelica.int, constant>

model Test
    constant Integer n;
end Test;
