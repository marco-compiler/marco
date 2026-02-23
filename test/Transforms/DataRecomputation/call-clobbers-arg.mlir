// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

module {
  func.func private @external_func(%arg: memref<i32>)

  func.func @test() {
    %alloc = memref.alloc() : memref<i32>
    %c42 = arith.constant 42 : i32
    memref.store %c42, %alloc[] : memref<i32>
    func.call @external_func(%alloc) : (memref<i32>) -> ()
    // expected-remark @below {{load: LEAKED}}
    %val = memref.load %alloc[] : memref<i32>
    memref.dealloc %alloc : memref<i32>
    return
  }
}
