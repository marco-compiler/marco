// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

module {
  func.func @test(%cond: i1) {
    %alloc = memref.alloc() : memref<i32>
    %val1 = arith.constant 1 : i32
    %val2 = arith.constant 2 : i32
    memref.store %val1, %alloc[] : memref<i32>
    scf.while : () -> () {
      scf.condition(%cond)
    } do {
      memref.store %val2, %alloc[] : memref<i32>
      scf.yield
    }
    // expected-remark @below {{load: MULTI}}
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
