// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-runtime | FileCheck %s

// CHECK: runtime.model_name "Test"

bmodelica.model @Test {

}
