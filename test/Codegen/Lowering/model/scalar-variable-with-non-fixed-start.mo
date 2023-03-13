// RUN: marco %s --omc-bypass --emit-mlir -o %t
// RUN: cat %t | FileCheck %s

// CHECK-LABEL: @Test
// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.real<5.000000e+00>
// CHECK-NEXT:      modelica.yield %[[value]] : !modelica.real
// CHECK-NEXT:  } {each = false, fixed = false}

model Test
    Real x(start = 5, fixed = false);
end Test;
