// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

// CHECK-LABEL: @R
// CHECK: modelica.variable @x : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @y : !modelica.variable<!modelica.real>

// CHECK-LABEL: @Test
// CHECK: modelica.variable @r.x : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @r.y : !modelica.variable<!modelica.real>

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.function @Test {
    modelica.variable @r : !modelica.variable<!modelica.record<@R>>
}

// -----

// Nested records.

// CHECK-LABEL: @R1
// CHECK: modelica.variable @x : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @y : !modelica.variable<!modelica.real>

// CHECK-LABEL: @R2
// CHECK: modelica.variable @r1.x : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @r1.y : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @r2.x : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @r2.y : !modelica.variable<!modelica.real>

// CHECK-LABEL: @Test
// CHECK: modelica.variable @r.r1.x : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @r.r1.y : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @r.r2.x : !modelica.variable<!modelica.real>
// CHECK: modelica.variable @r.r2.y : !modelica.variable<!modelica.real>

modelica.record @R1 {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.record @R2 {
    modelica.variable @r1 : !modelica.variable<!modelica.record<@R1>>
    modelica.variable @r2 : !modelica.variable<!modelica.record<@R1>>
}

modelica.function @Test {
    modelica.variable @r : !modelica.variable<!modelica.record<@R2>>
}

// -----

// Array of records.

// CHECK-LABEL: @Test
// CHECK: modelica.variable @r.x : !modelica.variable<3x4x5x!modelica.real>
// CHECK: modelica.variable @r.y : !modelica.variable<3x4x5x!modelica.real>

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.function @Test {
    modelica.variable @r : !modelica.variable<3x4x5x!modelica.record<@R>>
}

// -----

// Record composed by arrays.

// CHECK-LABEL: @Test
// CHECK: modelica.variable @r.x : !modelica.variable<3x!modelica.real>
// CHECK: modelica.variable @r.y : !modelica.variable<4x!modelica.real>

modelica.record @R {
    modelica.variable @x : !modelica.variable<3x!modelica.real>
    modelica.variable @y : !modelica.variable<4x!modelica.real>
}

modelica.function @Test {
    modelica.variable @r : !modelica.variable<!modelica.record<@R>>
}
