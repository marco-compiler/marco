// RUN: modelica-opt %s --split-input-file --one-shot-bufferize | FileCheck %s

// CHECK: runtime.function private @foo(memref<3xf64>)

runtime.function private @foo(tensor<3xf64>) -> ()

// -----

// CHECK: runtime.function private @foo(memref<?xf64>)

runtime.function private @foo(tensor<?xf64>) -> ()

// -----

// CHECK: runtime.function private @foo(memref<*xf64>)

runtime.function private @foo(tensor<*xf64>) -> ()
