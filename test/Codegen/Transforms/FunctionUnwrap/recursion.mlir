// RUN: modelica-opt %s --split-input-file --function-unwrap | FileCheck %s

// CHECK:       modelica.function @M1_M2_foo {
// CHECK-NEXT:      modelica.algorithm {
// CHECK-NEXT:          modelica.call @M1_M2_foo() : () -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:  }

modelica.model @M1 {
    modelica.model @M2 {
        modelica.function @foo {
            modelica.algorithm {
                modelica.call @M1::@M2::@foo() : () -> ()
            }
        }
    }
}
