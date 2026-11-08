// RUN: marco-opt %s --canonicalize --cse | FileCheck %s

// Constants materialized by the folder stay within the equation, statement
// or modification owning the expression, instead of being hoisted into the
// class body or out of the class.

// CHECK-LABEL: modelica.class model @M
// CHECK-NEXT:      modelica.component @x : @Real -> !modelica.real {
// CHECK-NEXT:          %[[c0:.*]] = modelica.constant #modelica<real 3.000000e+00> : !modelica.real
// CHECK-NEXT:          modelica.bind %[[c0]] : !modelica.real
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.equations {
// CHECK-NEXT:          modelica.for_equation %{{.*}} : !modelica.int in {
// CHECK-NEXT:              %[[c1:.*]] = modelica.constant #modelica<array [1, 2]> : tensor<2x!modelica.int>
// CHECK-NEXT:              modelica.yield %[[c1]]
// CHECK-NEXT:          } do {
// CHECK-NEXT:              modelica.equation {
// CHECK-NEXT:                  %[[c2:.*]] = modelica.constant #modelica<real 3.000000e+00> : !modelica.real
// CHECK-NEXT:                  %[[x:.*]] = modelica.ref @x
// CHECK-NEXT:                  modelica.equation_sides %[[x]] = %[[c2]]
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }

modelica.class model @M {
    modelica.component @x : @Real -> !modelica.real {
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        %1 = modelica.constant #modelica<real 2.0> : !modelica.real
        %2 = modelica.add %0, %1 : (!modelica.int, !modelica.real) -> !modelica.real
        modelica.bind %2 : !modelica.real
    }
    modelica.equations {
        modelica.for_equation %i : !modelica.int in {
            %0 = modelica.constant #modelica<int 1> : !modelica.int
            %1 = modelica.constant #modelica<int 2> : !modelica.int
            %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<2x!modelica.int>
            modelica.yield %2 : tensor<2x!modelica.int>
        } do {
            modelica.equation {
                %x = modelica.ref @x : !modelica.real
                %0 = modelica.constant #modelica<int 1> : !modelica.int
                %1 = modelica.constant #modelica<real 2.0> : !modelica.real
                %2 = modelica.add %0, %1 : (!modelica.int, !modelica.real) -> !modelica.real
                modelica.equation_sides %x = %2 : !modelica.real = !modelica.real
            }
        }
    }
}
