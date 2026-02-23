// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Load from alloc with no prior stores -- no provenance tracked.

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<i32>
    // No remark expected: no stores before this load
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
