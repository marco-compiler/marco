// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Multiple indexed stores, then a rank-0 store (via reinterpret_cast to scalar).
// The rank-0 store kills all prior indexed stores.

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<32xi32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %idx0 = arith.constant 0 : index
    %idx1 = arith.constant 1 : index
    memref.store %c1, %alloc[%idx0] : memref<32xi32>
    memref.store %c2, %alloc[%idx1] : memref<32xi32>
    // Rank-0 store via scalar view kills all prior indexed stores
    %scalar_view = memref.reinterpret_cast %alloc to offset: [0], sizes: [], strides: []
        : memref<32xi32> to memref<i32>
    memref.store %c3, %scalar_view[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %val = memref.load %alloc[%idx0] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
