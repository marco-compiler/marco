// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Load first, then store. The load is unaffected by the subsequent store.

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    // No remark expected: store hasn't happened yet
    %val = memref.load %alloc[] : memref<i32>
    memref.store %c42, %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
