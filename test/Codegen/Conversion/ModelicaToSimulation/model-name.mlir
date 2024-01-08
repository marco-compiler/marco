// RUN: modelica-opt %s --split-input-file --convert-modelica-to-simulation | FileCheck %s

// CHECK: simulation.model_name "Test"

modelica.model @Test {

}
