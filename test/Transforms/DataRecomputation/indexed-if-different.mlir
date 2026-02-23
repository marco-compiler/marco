// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// Store at index 0, then if-else stores at index 1 only.
// Load from index 0 should still be SINGLE (the original store).

module {
  func.func @test(%cond: i1) {
    %alloc = memref.alloc() : memref<32xi32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %idx0 = arith.constant 0 : index
    %idx1 = arith.constant 1 : index
    memref.store %c1, %alloc[%idx0] : memref<32xi32>
    scf.if %cond {
      memref.store %c2, %alloc[%idx1] : memref<32xi32>
    } else {
      memref.store %c3, %alloc[%idx1] : memref<32xi32>
    }
    // expected-remark @below {{load: SINGLE}}
    %val = memref.load %alloc[%idx0] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
