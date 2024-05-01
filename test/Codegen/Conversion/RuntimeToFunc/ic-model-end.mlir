// RUN: modelica-opt %s --split-input-file --convert-runtime-to-func | FileCheck %s

// CHECK:       func.func @icModelEnd() {
// CHECK-NEXT:      call @foo() : () -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:  }

func.func @foo() {
    func.return
}

runtime.ic_model_end {
    func.call @foo() : () -> ()
}

// -----

// CHECK:       func.func @icModelEnd() {
// CHECK-NEXT:      call @foo() : () -> ()
// CHECK-NEXT:      call @bar() : () -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:  }

func.func @foo() {
    func.return
}

func.func @bar() {
    func.return
}

runtime.ic_model_end {
    func.call @foo() : () -> ()
}

runtime.ic_model_end {
    func.call @bar() : () -> ()
}
