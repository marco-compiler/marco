// RUN: modelica-opt %s --split-input-file --function-mangling | FileCheck %s

// CHECK: bmodelica.function @_Mfoo
// CHECK-NOT: bmodelica.function @foo

bmodelica.function @foo {

}
