// RUN: modelica-opt %s --split-input-file --one-shot-bufferize | FileCheck %s

// CHECK: runtime.function private @staticTensorArg(memref<3xf64>)

runtime.function private @staticTensorArg(tensor<3xf64>) -> ()

// -----

// CHECK: runtime.function private @dynamicTensorArg(memref<?xf64>)

runtime.function private @dynamicTensorArg(tensor<?xf64>) -> ()

// -----

// CHECK: runtime.function private @unrankedTensorArg(memref<*xf64>)

runtime.function private @unrankedTensorArg(tensor<*xf64>) -> ()
