// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<32xi32>
    %c42 = arith.constant 42 : i32
    %idx = arith.constant 0 : index
    %sv = memref.subview %alloc[0][16][1] : memref<32xi32> to memref<16xi32>
    memref.store %c42, %sv[%idx] : memref<16xi32>
    // expected-remark @below {{load: SINGLE}}
    %val = memref.load %alloc[%idx] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
