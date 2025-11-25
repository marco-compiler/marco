// RUN: modelica-opt %s --split-input-file --access-affine-raise --canonicalize | FileCheck %s

// CHECK-LABEL: @OneLoop
// CHECK-SAME:  (%[[arg0:.*]]: memref<10xf64>)
// CHECK:   affine.for %[[i:.*]] = 0 to 9 {
// CHECK:       %[[load:.*]] = affine.load %[[arg0]][%[[i]]]
// CHECK:       affine.store %[[load]], %[[arg0]][%[[i]]]
// CHECK:   }

func.func @OneLoop(%arg0: memref<10xf64>) {
    affine.for %i = 0 to 9 {
        %0 = memref.load %arg0[%i] : memref<10xf64>
        memref.store %0, %arg0[%i] : memref<10xf64>
    }

    func.return
}

// -----

// CHECK-LABEL: @NestedLoops
// CHECK-SAME:  (%[[arg0:.*]]: memref<10x20x30xf64>)
// CHECK:   affine.for %[[i:.*]] = 0 to 9 {
// CHECK:       affine.for %[[j:.*]] = 0 to 19 {
// CHECK:           affine.for %[[k:.*]] = 0 to 29 {
// CHECK:               %[[load:.*]] = affine.load %[[arg0]][%[[i]], %[[j]], %[[k]]]
// CHECK:               affine.store %[[load]], %[[arg0]][%[[i]], %[[j]], %[[k]]]
// CHECK:           }
// CHECK:       }
// CHECK:   }

func.func @NestedLoops(%arg0: memref<10x20x30xf64>) {
    affine.for %i = 0 to 9 {
        affine.for %j = 0 to 19 {
            affine.for %k = 0 to 29 {
                %0 = memref.load %arg0[%i, %j, %k] : memref<10x20x30xf64>
                memref.store %0, %arg0[%i, %j, %k] : memref<10x20x30xf64>
            }
        }
    }

  func.return
}
