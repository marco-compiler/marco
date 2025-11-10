// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' | FileCheck %s

// CHECK: Write: memref.store %c1_i32, %[[memref:.*]][%c1] : memref<32xi32>
// CHECK-NEXT: Load:
// CHECK-SAME: memref.load{{.*}}
// CHECK-NEXT: Origin Allocation
// CHECK-SAME: memref.global @mystuff{{.*}}

module {
  memref.global @mystuff : memref<32xi32> = uninitialized

  func.func @consumes_memref(%idx: i32, %memref: memref<32xi32>) -> i32 {
    %cst1 = arith.constant 0 : i32
    %idxt = arith.index_cast %idx : i32 to index
    %val = arith.addi %idx, %cst1 : i32

    %reinterpret_cast = memref.reinterpret_cast %memref to
        offset: [2], sizes: [32], strides: [0]
        : memref<32xi32> to memref<32xi32, strided<[0], offset: 2>>

    %loaded_val = memref.load %reinterpret_cast[%idxt] : memref<32xi32, strided<[0], offset: 2>>

    return %cst1 : i32
  }

  func.func @main() attributes{} {
    %alloc = memref.get_global @mystuff : memref<32xi32>
    %c1 = arith.constant 1 : i32
    %idx = arith.constant 1 : index
    memref.store %c1, %alloc[%idx] : memref<32xi32>

    %res = func.call @consumes_memref(%c1, %alloc) : (i32, memref<32xi32>) -> i32

    return
  }
}
