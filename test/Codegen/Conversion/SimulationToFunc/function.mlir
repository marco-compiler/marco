// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// No solvers and no variables

// CHECK:       func.func @foo(%[[runtimeDataOpaquePtr:.*]]: !llvm.ptr<i8>) {
// CHECK:           return
// CHECK-NEXT:  }

simulation.module {
    simulation.function @foo(solvers: [], time: [%time : f64], variables: [], extra_args: []) -> () {
        simulation.return
    }
}

// -----

// Usage of time variable

// CHECK:       func.func @foo(%[[runtimeDataOpaquePtr:.*]]: !llvm.ptr<i8>) -> f64 {
// CHECK:           %[[runtimeDataPtr:.*]] = llvm.bitcast %[[runtimeDataOpaquePtr]] : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i8>, f64)>>
// CHECK:           %[[runtimeData:.*]] = llvm.load %[[runtimeDataPtr]]
// CHECK:           %[[time:.*]] = llvm.extractvalue %[[runtimeData]][1]
// CHECK:           return %[[time]]
// CHECK-NEXT:  }

simulation.module {
    simulation.function @foo(solvers: [], time: [%time : f64], variables: [], extra_args: []) -> (f64) {
        simulation.return %time : f64
    }
}

// -----

// Usage of solvers

// CHECK:       func.func @foo(%[[runtimeDataOpaquePtr:.*]]: !llvm.ptr<i8>) -> i64 {
// CHECK:           %[[runtimeDataPtr:.*]] = llvm.bitcast %[[runtimeDataOpaquePtr]] : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i8>, f64)>>
// CHECK:           %[[runtimeData:.*]] = llvm.load %[[runtimeDataPtr]]
// CHECK:           %[[solversOpaquePtr:.*]] = llvm.extractvalue %[[runtimeData]][0]
// CHECK:           %[[solversPtr:.*]] = llvm.bitcast %[[solversOpaquePtr]] : !llvm.ptr<i8> to !llvm.ptr<struct<(i64)>>
// CHECK:           %[[solvers:.*]] = llvm.load %[[solversPtr]]
// CHECK:           %[[solver:.*]] = llvm.extractvalue %[[solvers]][0]
// CHECK:           return %[[solver]]
// CHECK-NEXT:  }

simulation.module {
    simulation.function @foo(solvers: [%solver0: i64], time: [%arg0 : f64], variables: [], extra_args: []) -> (i64) {
        simulation.return %solver0 : i64
    }
}
