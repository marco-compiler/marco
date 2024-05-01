// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.model @M {
// CHECK-NEXT:      bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @Test::@R>>
// CHECK-NEXT:  }

package Test
    record R
    end R;

    model M
        R r;
    end M;
end Test;
