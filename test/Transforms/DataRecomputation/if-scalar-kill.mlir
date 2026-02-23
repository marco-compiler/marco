// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Rank-0 store before if, rank-0 in both branches.
// The prior store is killed in both branches; two branch stores survive.

module {
  func.func @test(%cond: i1) {
    %alloc = memref.alloc() : memref<i32>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    memref.store %c0, %alloc[] : memref<i32>
    scf.if %cond {
      memref.store %c1, %alloc[] : memref<i32>
    } else {
      memref.store %c2, %alloc[] : memref<i32>
    }
    // expected-remark @below {{load: MULTI}}
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
