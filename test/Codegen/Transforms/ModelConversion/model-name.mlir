// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(test-model-conversion{model=Test})" | FileCheck %s

// CHECK: simulation.module
// CHECK: modelName = "Test"

modelica.model @Test {

} body {

}
