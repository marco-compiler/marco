// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A call with scalar-only args does not clobber a local allocation.

module {
  func.func private @scalar_func(%x: i32) -> i32

  func.func @test() {
    %alloc = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    memref.store %c42, %alloc[] : memref<i32>
    %result = func.call @scalar_func(%c42) : (i32) -> i32
    // expected-remark @below {{load: SINGLE}}
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
