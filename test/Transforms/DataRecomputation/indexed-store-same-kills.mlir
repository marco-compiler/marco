// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Two stores to the same constant index: the second kills the first.

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<32xi32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %idx = arith.constant 5 : index
    memref.store %c1, %alloc[%idx] : memref<32xi32>
    memref.store %c2, %alloc[%idx] : memref<32xi32>
    // expected-remark @below {{load: SINGLE}}
    %val = memref.load %alloc[%idx] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
