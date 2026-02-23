// RUN: modelica-opt %s --pass-pipeline='builtin.module(data-recomputation{dr-test-diagnostics})' -verify-diagnostics

// A store with a dynamic (non-constant) index falls back to conservative
// behavior: can't determine offset, so any subsequent load reports MULTI.

module {
  func.func @test(%dyn_idx: index) {
    %alloc = memref.alloc() : memref<32xi32>
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %idx0 = arith.constant 0 : index
    memref.store %c1, %alloc[%idx0] : memref<32xi32>
    memref.store %c2, %alloc[%dyn_idx] : memref<32xi32>
    // expected-remark @below {{load: MULTI}}
    %val = memref.load %alloc[%idx0] : memref<32xi32>
    memref.dealloc %alloc : memref<32xi32>
    return
  }
}
