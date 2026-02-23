// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Alloc A, B. Load A -> store to B. Overwrite A.
// The store to B is killed because its value depended on A.
// After kill, the load from B sees an empty provenance set, classified as KILLED.

module {
  func.func @test() {
    %a = memref.alloc() : memref<i32>
    %b = memref.alloc() : memref<i32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    memref.store %c1, %a[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %loaded = memref.load %a[] : memref<i32>
    memref.store %loaded, %b[] : memref<i32>
    memref.store %c2, %a[] : memref<i32>
    // expected-remark @below {{load: KILLED}}
    %val = memref.load %b[] : memref<i32>
    memref.dealloc %a : memref<i32>
    memref.dealloc %b : memref<i32>
    return
  }
}
