// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s --check-prefix="CHECK-DEFAULT"
// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-simulation-to-func{emit-main-function=true})" | FileCheck %s --check-prefix="CHECK-ENABLED"
// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-simulation-to-func{emit-main-function=false})" | FileCheck %s --check-prefix="CHECK-DISABLED"

// Default behaviour.

// CHECK-DEFAULT:       func.func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
// CHECK-DEFAULT-NEXT:      %[[result:.*]] = llvm.call @runSimulation(%arg0, %arg1) : (i32, !llvm.ptr<ptr<i8>>) -> i32
// CHECK-DEFAULT-NEXT:      return %[[result]] : i32
// CHECK-DEFAULT-NEXT:  }

simulation.module {

}

// -----

// Explicitly enabled main function.

// CHECK-ENABLED:       func.func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
// CHECK-ENABLED-NEXT:      %[[result:.*]] = llvm.call @runSimulation(%arg0, %arg1) : (i32, !llvm.ptr<ptr<i8>>) -> i32
// CHECK-ENABLED-NEXT:      return %[[result]] : i32
// CHECK-ENABLED-NEXT:  }

simulation.module {

}

// -----

// Explicitly disabled main function.

// CHECK-DISABLED-NOT: @main

simulation.module {

}
