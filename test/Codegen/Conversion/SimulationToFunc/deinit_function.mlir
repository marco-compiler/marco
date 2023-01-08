// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// No variables

// CHECK:       func.func @deinit(%[[opaquePtr:.*]]: !llvm.ptr<i8>) {
// CHECK-NEXT:      %[[structPtr:.*]] = llvm.bitcast %[[opaquePtr]] : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i8>, f64)>>
// CHECK-NEXT:      %[[structValue:.*]] = llvm.load %[[structPtr]]
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      return
// CHECK-NEXT:  }

simulation.module {
    simulation.deinit_function() {
        simulation.yield
    }
}

// -----

// Generic case

// CHECK:       func.func @deinit(%[[opaquePtr:.*]]: !llvm.ptr<i8>) {
// CHECK-NEXT:      %[[structPtr:.*]] = llvm.bitcast %[[opaquePtr]] : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i8>, f64, f64, f64)>>
// CHECK-NEXT:      %[[structValue:.*]] = llvm.load %[[structPtr]]
// CHECK-NEXT:      %[[arg0:.*]] = llvm.extractvalue %[[structValue]][2] : !llvm.struct<(ptr<i8>, f64, f64, f64)>
// CHECK-NEXT:      %[[arg1:.*]] = llvm.extractvalue %[[structValue]][3] : !llvm.struct<(ptr<i8>, f64, f64, f64)>
// CHECK-NEXT:     cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:
// CHECK-NEXT:      %{{.*}} = arith.addf %[[arg0]], %[[arg1]] : f64
// CHECK-NEXT:      return
// CHECK-NEXT:  }

simulation.module {
    simulation.deinit_function(%arg0: f64, %arg1: f64) {
        %0 = arith.addf %arg0, %arg1 : f64
        simulation.yield
    }
}
