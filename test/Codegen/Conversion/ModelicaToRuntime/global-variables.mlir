// RUN: modelica-opt %s --split-input-file --convert-modelica-to-runtime | FileCheck %s

// CHECK-DAG: bmodelica.global_variable @[[x:.*]] : !bmodelica.array<!bmodelica.int>
// CHECK-DAG: bmodelica.global_variable @[[y:.*]] : !bmodelica.array<3x!bmodelica.int>

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
        bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int>
    }
}
