// RUN: modelica-opt %s --split-input-file --pass-pipeline="convert-model{model=Test}" | FileCheck %s

// CHECK:   func.func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
// CHECK:       %[[result:.*]] = llvm.call @runSimulation(%arg0, %arg1) : (i32, !llvm.ptr<ptr<i8>>) -> i32
// CHECK:       return %[[result]] : i32
// CHECK:   }

modelica.model @Test {

} body {

}

// -----

// RUN: modelica-opt %s --split-input-file --pass-pipeline="convert-model{model=Test emit-simulation-main-function=false}" | FileCheck %s --check-prefix="CHECK-DISABLED"

// CHECK-DISABLED-NOT: @main

modelica.model @Test {

} body {

}
