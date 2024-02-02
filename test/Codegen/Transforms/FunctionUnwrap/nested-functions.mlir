// RUN: modelica-opt %s --split-input-file --function-unwrap | FileCheck %s

// CHECK:       modelica.model @M1 {
// CHECK-NEXT:  }
// CHECK-NEXT:  modelica.function @M1_foo {
// CHECK-NEXT:      modelica.algorithm {
// CHECK-NEXT:          modelica.call @M1_foo_bar() : () -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:  }
// CHECK-NEXT:  modelica.function @M1_foo_bar {
// CHECK-NEXT:  }

modelica.model @M1 {
    modelica.function @foo {
        modelica.function @bar {

        }

        modelica.algorithm {
            modelica.call @M1::@foo::@bar() : () -> ()
        }
    }
}
