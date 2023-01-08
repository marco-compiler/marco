// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// One solver

// CHECK:       func.func @initICSolvers(%[[runtimeDataOpaquePtr:.*]]: !llvm.ptr<i8>) {
// CHECK:           %[[runtimeDataPtr:.*]] = llvm.bitcast %[[runtimeDataOpaquePtr]] : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i8>, f64)>>
// CHECK:           %[[runtimeData:.*]] = llvm.load %[[runtimeDataPtr]]
// CHECK:           cf.br ^[[mainBlock:.*]]
// CHECK-NEXT:  ^[[mainBlock]]:
// CHECK:           %[[solver:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[solversDataUndef:.*]] = llvm.mlir.undef : !llvm.struct<(f64)>
// CHECK:           %[[solversData:.*]] = llvm.insertvalue %cst, %[[solversDataUndef]][0]
// CHECK:           %[[solversDataOpaquePtr:.*]] = llvm.call @_MheapAlloc_pvoid_i64(%{{.*}}) : (i64) -> !llvm.ptr<i8>
// CHECK:           %[[solversDataPtr:.*]] = llvm.bitcast %[[solversDataOpaquePtr]] : !llvm.ptr<i8> to !llvm.ptr<struct<(f64)>>
// CHECK:           llvm.store %[[solversData]], %[[solversDataPtr]]
// CHECK:           %[[solversDataOpaquePtr:.*]] = llvm.bitcast %[[solversDataPtr]] : !llvm.ptr<struct<(f64)>> to !llvm.ptr<i8>
// CHECK:           %[[newRuntimeData:.*]] = llvm.insertvalue %[[solversDataOpaquePtr]], %[[runtimeData]][0]
// CHECK:           %[[runtimeDataPtr:.*]] = llvm.bitcast %[[runtimeDataOpaquePtr]] : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i8>, f64)>>
// CHECK:           llvm.store %[[newRuntimeData]], %[[runtimeDataPtr]]
// CHECK:           return
// CHECK-NEXT:  }

simulation.module {
    simulation.init_ic_solvers_function () -> (f64) {
        %0 = arith.constant 0.0 : f64
        simulation.yield %0 : f64
    }
}
