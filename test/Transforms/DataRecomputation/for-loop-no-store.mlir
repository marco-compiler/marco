// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

module {
  func.func @test() {
    %alloc = memref.alloc() : memref<i32>
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1_idx = arith.constant 1 : index
    %val1 = arith.constant 1 : i32
    memref.store %val1, %alloc[] : memref<i32>
    scf.for %i = %c0 to %c10 step %c1_idx {
      // No store in loop body
    }
    // expected-remark @below {{load: SINGLE}}
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
