// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Store a constant (no allocation dependency), load it back.
// The store is not killed by any overwrite because constants have no deps.

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    memref.store %c42, %alloc[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
