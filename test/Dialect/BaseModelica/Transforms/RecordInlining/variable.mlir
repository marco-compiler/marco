// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

// CHECK-LABEL: @recordOfScalars

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.function @recordOfScalars {
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>>

    // CHECK: bmodelica.variable @r.x : !bmodelica.variable<!bmodelica.real>
    // CHECK: bmodelica.variable @r.y : !bmodelica.variable<!bmodelica.real>
}

// -----

// CHECK-LABEL: @nestedRecords

bmodelica.record @R1 {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.record @R2 {
    bmodelica.variable @r1 : !bmodelica.variable<!bmodelica<record @R1>>
    bmodelica.variable @r2 : !bmodelica.variable<!bmodelica<record @R1>>
}

bmodelica.function @nestedRecords {
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R2>>

    // CHECK: bmodelica.variable @r.r1.x : !bmodelica.variable<!bmodelica.real>
    // CHECK: bmodelica.variable @r.r1.y : !bmodelica.variable<!bmodelica.real>
    // CHECK: bmodelica.variable @r.r2.x : !bmodelica.variable<!bmodelica.real>
    // CHECK: bmodelica.variable @r.r2.y : !bmodelica.variable<!bmodelica.real>
}

// -----

// CHECK-LABEL: @arrayOfRecords

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.function @arrayOfRecords {
    bmodelica.variable @r : !bmodelica.variable<3x4x5x!bmodelica<record @R>>

    // CHECK-DAG: bmodelica.variable @r.x : !bmodelica.variable<3x4x5x!bmodelica.real>
    // CHECK-DAG: bmodelica.variable @r.y : !bmodelica.variable<3x4x5x!bmodelica.real>
}

// -----

// CHECK-LABEL: @recordOfArrays

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<4x!bmodelica.real>
}

bmodelica.function @recordOfArrays {
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>>

    // CHECK-DAG: bmodelica.variable @r.x : !bmodelica.variable<3x!bmodelica.real>
    // CHECK-DAG: bmodelica.variable @r.y : !bmodelica.variable<4x!bmodelica.real>
}
