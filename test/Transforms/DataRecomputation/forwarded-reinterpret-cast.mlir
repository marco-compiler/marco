// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Same as forwarded-memref-arg but callee does reinterpret_cast before load.
// Verifies view chain tracing on block arg.

module {
  memref.global "private" @g : memref<32xi32> = uninitialized

  func.func @callee(%memref: memref<32xi32>) -> i32 {
    %idx = arith.constant 0 : index
    %rc = memref.reinterpret_cast %memref to offset: [0], sizes: [32], strides: [1]
        : memref<32xi32> to memref<32xi32, strided<[1], offset: 0>>
    // No remark expected: view chain traces to block arg, no tracked provenance
    %val = memref.load %rc[%idx] : memref<32xi32, strided<[1], offset: 0>>
    return %val : i32
  }

  func.func @caller() {
    %ref = memref.get_global @g : memref<32xi32>
    %c42 = arith.constant 42 : i32
    %idx = arith.constant 0 : index
    memref.store %c42, %ref[%idx] : memref<32xi32>
    %result = func.call @callee(%ref) : (memref<32xi32>) -> i32
    // expected-remark @below {{load: SINGLE}}
    %val = memref.load %ref[%idx] : memref<32xi32>
    return
  }
}
