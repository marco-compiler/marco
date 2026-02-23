// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Caller stores to global, passes it as arg to callee.
// Callee loads via the arg -- analysis is per-function, callee sees block arg
// with no tracked provenance.

module {
  memref.global "private" @g : memref<32xi32> = uninitialized

  func.func @callee(%memref: memref<32xi32>) -> i32 {
    %idx = arith.constant 0 : index
    // No remark expected: per-function analysis, block arg has no tracked provenance
    %val = memref.load %memref[%idx] : memref<32xi32>
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
