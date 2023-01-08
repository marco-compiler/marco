// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// One solver

// CHECK:       func.func @deinitMainSolvers(%[[runtimeDataOpaquePtr:.*]]: !llvm.ptr<i8>) {
// CHECK:           %[[runtimeDataPtr:.*]] = llvm.bitcast %[[runtimeDataOpaquePtr]] : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i8>, f64)>>
// CHECK:           %[[runtimeData:.*]] = llvm.load %[[runtimeDataPtr]]
// CHECK:           %[[solversOpaquePtr:.*]] = llvm.extractvalue %[[runtimeData]][0]
// CHECK:           %[[solversPtr:.*]] = llvm.bitcast %[[solversOpaquePtr]] : !llvm.ptr<i8> to !llvm.ptr<struct<(f64)>>
// CHECK:           %[[solvers:.*]] = llvm.load %[[solversPtr]]
// CHECK:           %[[solver:.*]] = llvm.extractvalue %[[solvers]][0]
// CHECK-NEXT:      cf.br ^[[mainBlock:.*]]
// CHECK-NEXT:  ^[[mainBlock]]:
// CHECK:           %[[cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           arith.addf %[[solver]], %[[cst]]
// CHECK:           return
// CHECK-NEXT:  }

simulation.module {
    simulation.deinit_main_solvers_function (%arg0: f64) {
        %0 = arith.constant 0.0 : f64
        %1 = arith.addf %arg0, %0 : f64
        simulation.yield
    }
}
