// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<64xi32>
    %c42 = arith.constant 42 : i32
    %idx = arith.constant 0 : index
    %sv = memref.subview %alloc[0][32][1] : memref<64xi32> to memref<32xi32>
    %rc = memref.reinterpret_cast %sv to offset: [0], sizes: [32], strides: [1]
        : memref<32xi32> to memref<32xi32, strided<[1], offset: 0>>
    memref.store %c42, %rc[%idx] : memref<32xi32, strided<[1], offset: 0>>
    // expected-remark @below {{load: SINGLE}}
    %val = memref.load %alloc[%idx] : memref<64xi32>
    memref.dealloc %alloc : memref<64xi32>
    return
  }
}
