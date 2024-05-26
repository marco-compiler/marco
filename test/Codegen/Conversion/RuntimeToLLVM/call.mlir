// RUN: modelica-opt %s --split-input-file --convert-runtime-to-llvm | FileCheck %s

// CHECK-LABEL: @bar
// CHECK: %[[stackPtr:.*]] = llvm.intr.stacksave
// CHECK: llvm.call @foo()
// CHECK: llvm.intr.stackrestore %[[stackPtr]]

runtime.function private @foo()

func.func @bar() {
    runtime.call @foo() : () -> ()
    func.return
}

// -----

// CHECK-LABEL: @bar
// CHECK-SAME: (%[[arg0:.*]]: f64)
// CHECK: %[[stackPtr:.*]] = llvm.intr.stacksave
// CHECK: llvm.call @foo(%[[arg0]])
// CHECK: llvm.intr.stackrestore %[[stackPtr]]

runtime.function private @foo(f64)

func.func @bar(%arg0: f64) {
    runtime.call @foo(%arg0) : (f64) -> ()
    func.return
}

// -----

// CHECK-LABEL: @bar
// CHECK: %[[stackPtr:.*]] = llvm.intr.stacksave
// CHECK: %[[alloca:.*]] = llvm.alloca %{{.*}} x !llvm.struct<(i64, ptr)>
// CHECK: llvm.store %{{.*}}, %[[alloca]]
// CHECK: llvm.call @foo(%[[alloca]])
// CHECK: llvm.intr.stackrestore %[[stackPtr]]

runtime.function private @foo(memref<*xf64>)

func.func @bar(%arg0: memref<*xf64>) {
    runtime.call @foo(%arg0) : (memref<*xf64>) -> ()
    func.return
}

// -----

// CHECK-LABEL: @bar
// CHECK: %[[stackPtr:.*]] = llvm.intr.stacksave
// CHECK: %[[alloca:.*]] = llvm.alloca %{{.*}} x !llvm.struct<(i64, ptr)>
// CHECK: llvm.store %{{.*}}, %[[alloca]]
// CHECK: llvm.call @foo(%[[alloca]])
// CHECK: llvm.intr.stackrestore %[[stackPtr]]

runtime.function private @foo(memref<?xf64>)

func.func @bar(%arg0: memref<?xf64>) {
    runtime.call @foo(%arg0) : (memref<?xf64>) -> ()
    func.return
}

// -----

// CHECK: llvm.func @foo(!llvm.ptr)

runtime.function private @foo(memref<3xf64>)

func.func @bar(%arg0: memref<3xf64>) {
    runtime.call @foo(%arg0) : (memref<3xf64>) -> ()
    func.return
}
