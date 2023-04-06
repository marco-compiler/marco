// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK:       modelica.model @M {
// CHECK-NEXT:      modelica.variable @r : !modelica.variable<!modelica.record<"R">>
// CHECK-NEXT:  }

package Test
    record R
    end R;

    model M
        R r;
    end M;
end Test;
