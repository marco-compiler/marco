// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// No variables

// CHECK:       func.func @init() -> !llvm.ptr<i8> {
// CHECK:           %[[structValue:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
// CHECK:           %[[structPtr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i8>, f64)>>
// CHECK:           llvm.store %[[structValue]], %[[structPtr]]
// CHECK:           %[[result:.*]] = llvm.bitcast %[[structPtr]] : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
// CHECK:           return %[[result]]
// CHECK-NEXT:  }

simulation.module {
    simulation.init_function () -> () {
        simulation.yield
    }
}

// -----

// Generic case

// CHECK:       func.func @init() -> !llvm.ptr<i8> {
// CHECK-DAG:       %[[var1:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[var2:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[structValue_1:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64, i64, f64)>
// CHECK:           %[[structValue_2:.*]] = llvm.insertvalue %[[var1]], %[[structValue_1]][2]
// CHECK:           %[[structValue_3:.*]] = llvm.insertvalue %[[var2]], %[[structValue_2]][3]
// CHECK:           %[[structPtr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i8>, f64, i64, f64)>>
// CHECK:           llvm.store %[[structValue_3]], %[[structPtr]]
// CHECK:           %[[result:.*]] = llvm.bitcast %[[structPtr]] : !llvm.ptr<struct<(ptr<i8>, f64, i64, f64)>> to !llvm.ptr<i8>
// CHECK:           return %[[result]]
// CHECK-NEXT:  }

simulation.module {
    simulation.init_function () -> (i64, f64) {
        %0 = arith.constant 0 : i64
        %1 = arith.constant 0.0 : f64
        simulation.yield %0, %1 : i64, f64
    }
}
