// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<i32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    memref.store %c1, %alloc[] : memref<i32>
    affine.if affine_set<() : (0 >= 0)>() {
      memref.store %c2, %alloc[] : memref<i32>
    }
    // expected-remark @below {{load: MULTI}}
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
