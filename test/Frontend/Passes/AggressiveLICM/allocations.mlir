// RUN: modelica-opt %s --split-input-file --aggressive-licm | FileCheck %s

// CHECK-LABEL: @AllocationInsideInnermostLoop
// CHECK-DAG:   %[[alloca:.*]] = memref.alloca
// CHECK-DAG:   %[[alloc:.*]] = memref.alloc
// CHECK:       memref.load
// CHECK:       affine.for
// CHECK:       affine.for
// CHECK-NOT:   memref.alloc

func.func @AllocationInsideInnermostLoop() -> f64 {
    %alloca = memref.alloca() : memref<f64>

    affine.for %i = 0 to 10 {
        affine.for %j = 0 to 10 {
            %alloc = memref.alloc() : memref<f64>
            %val = memref.load %alloc[] : memref<f64>
            memref.store %val, %alloca[] : memref<f64>
        }
    }

    %res = memref.load %alloca[] : memref<f64>
    func.return %res : f64
}

// -----

// CHECK-LABEL: @UnhoistableAllocation
// CHECK:   %[[alloca:.*]] = memref.alloca
// CHECK:   affine.for %[[i:.*]] = 1 to 10
// CHECK:       %[[alloc:.*]] = memref.alloc(%[[i]])

func.func @UnhoistableAllocation() -> f64 {
    %alloca = memref.alloca() : memref<f64>

    affine.for %i = 1 to 10 {
        %alloc = memref.alloc(%i) : memref<?xf64>
        %idx = affine.apply affine_map<(i)[] -> (i - 1)> (%i)[]
        %val = memref.load %alloc[%idx] : memref<?xf64>
        memref.store %val, %alloca[] : memref<f64>
    }

    %res = memref.load %alloca[] : memref<f64>
    func.return %res : f64
}

// -----

// CHECK-LABEL: @AllocationDependingOnOuterLoop
// CHECK:   %[[alloca:.*]] = memref.alloca
// CHECK:   affine.for %[[i:.*]] = 1 to 10
// CHECK:       %[[alloc:.*]] = memref.alloc(%[[i]])
// CHECK:       %[[idx:.*]] = affine.apply
// CHECK:       memref.load %[[alloc]][%[[idx]]]
// CHECK:       affine.for %[[j:.*]] = 0 to 9

func.func @AllocationDependingOnOuterLoop() -> f64 {
    %alloca = memref.alloca() : memref<10xf64>

    affine.for %i = 1 to 10 {
        affine.for %j = 0 to 9 {
            %alloc = memref.alloc(%i) : memref<?xf64>
            %idx = affine.apply affine_map<(i)[] -> (i - 1)> (%i)[]
            %val = memref.load %alloc[%idx] : memref<?xf64>
            memref.store %val, %alloca[%j] : memref<10xf64>
        }
    }

    %zero = arith.constant 0 : index
    %res = memref.load %alloca[%zero] : memref<10xf64>
    func.return %res : f64
}
