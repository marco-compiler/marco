// RUN: modelica-opt %s --split-input-file --test-model-conversion | FileCheck %s

// CHECK: simulation.model_name "Test"

modelica.model @Test {

}
