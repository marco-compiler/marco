// RUN: modelica-opt %s --split-input-file --one-shot-bufferize | FileCheck %s

// CHECK-LABEL: @staticTensorArg

runtime.function private @foo(tensor<3xf64>) -> ()

func.func @staticTensorArg() {
    %0 = tensor.empty() : tensor<3xf64>

    runtime.call @foo(%0) : (tensor<3xf64>) -> ()
    // CHECK: runtime.call @foo(%{{.*}}) : (memref<3xf64>) -> ()

    func.return
}

// -----

// CHECK-LABEL: @dynamicTensorArg

runtime.function private @foo(tensor<?xf64>) -> ()

func.func @dynamicTensorArg() {
    %0 = arith.constant 3 : index
    %1 = tensor.empty(%0) : tensor<?xf64>

    runtime.call @foo(%1) : (tensor<?xf64>) -> ()
    // CHECK: runtime.call @foo(%{{.*}}) : (memref<?xf64>) -> ()

    func.return
}

// -----

// CHECK-LABEL: @unrankedTensorArg

runtime.function private @foo(tensor<*xf64>) -> ()

func.func @unrankedTensorArg() {
    %0 = tensor.empty() : tensor<3xf64>
    %1 = tensor.cast %0 : tensor<3xf64> to tensor<*xf64>

    runtime.call @foo(%1) : (tensor<*xf64>) -> ()
    // CHECK: runtime.call @foo(%{{.*}}) : (memref<*xf64>) -> ()

    func.return
}
