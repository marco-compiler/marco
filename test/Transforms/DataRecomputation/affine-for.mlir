// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<i32>
    %val1 = arith.constant 1 : i32
    %val2 = arith.constant 2 : i32
    memref.store %val1, %alloc[] : memref<i32>
    affine.for %i = 0 to 10 {
      memref.store %val2, %alloc[] : memref<i32>
    }
    // expected-remark @below {{load: SINGLE}}
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
