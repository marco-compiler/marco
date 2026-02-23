// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

module {
  memref.global "private" @g : memref<i32> = uninitialized

  func.func @test() {
    %ref = memref.get_global @g : memref<i32>
    %c42 = arith.constant 42 : i32
    memref.store %c42, %ref[] : memref<i32>
    // expected-remark @below {{load: SINGLE}}
    %val = memref.load %ref[] : memref<i32>
    return
  }
}
