// RUN: marco %s --omc-bypass --emit-mlir -o %t
// RUN: cat %t | FileCheck %s

// CHECK-LABEL: @Test
// CHECK: modelica.member_create @n : !modelica.member<!modelica.int, constant>

model Test
    constant Integer n;
end Test;
