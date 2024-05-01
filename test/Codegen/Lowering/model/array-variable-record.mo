// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.model @M {
// CHECK-NEXT:      bmodelica.variable @r : !bmodelica.variable<3x!bmodelica<record @Test::@R>>
// CHECK-NEXT:  }

package Test
    record R
    end R;

    model M
        R[3] r;
    end M;
end Test;
