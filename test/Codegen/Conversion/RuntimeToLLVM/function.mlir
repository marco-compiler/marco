// RUN: modelica-opt %s --split-input-file --convert-runtime-to-llvm | FileCheck %s

// CHECK: llvm.func @foo()

runtime.function private @foo()

// -----

// CHECK: llvm.func @foo(f64)

runtime.function private @foo(f64)

// -----

// CHECK: llvm.func @foo(!llvm.ptr)

runtime.function private @foo(memref<*xf64>)

// -----

// CHECK: llvm.func @foo(!llvm.ptr)

runtime.function private @foo(memref<?xf64>)

// -----

// CHECK: llvm.func @foo(!llvm.ptr)

runtime.function private @foo(memref<3xf64>)
