// RUN: modelica-opt %s --split-input-file --convert-modelica-to-runtime | FileCheck %s

// CHECK: runtime.model_name "Test"

modelica.model @Test {

}
