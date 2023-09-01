// RUN: modelica-opt %s --split-input-file --test-model-conversion | FileCheck %s

// CHECK-DAG: modelica.global_variable @[[x:.*]] : !modelica.array<!modelica.int>
// CHECK-DAG: modelica.global_variable @[[y:.*]] : !modelica.array<3x!modelica.int>

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
    modelica.variable @y : !modelica.variable<3x!modelica.int>
}
