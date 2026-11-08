// RUN: marco-opt %s --modelica-instantiate=model=M | FileCheck %s

// Arrays of instances are instantiated once for all their elements. The
// modifications without 'each' are split among the elements,
// and the equations are repeated for each element by
// means of for-equations.
//
// model R
//   parameter Real r;
//   parameter Integer n = 2;
//   Real x[n];
// equation
//   x = fill(r, n);
// end R;
// model M
//   parameter Integer N = 3;
//   R rs[N](r = {1, 2, 3}, each n = 4);
//   Real y[N];
// equation
//   for i in 1:N loop
//     y[i] = rs[i].x[1];
//   end for;
// end M;

// CHECK:       modelica.model @M {
// CHECK:           modelica.instance model @rs : tensor<3x!modelica.object<@R>> {
// CHECK-NEXT:          modelica.variable parameter @r : !modelica.real {
// CHECK-NEXT:          ^bb0(%[[i:.*]]: !modelica.int):
// CHECK:                   %[[values:.*]] = modelica.array
// CHECK-NEXT:              %[[value:.*]] = modelica.subscript %[[values]][%[[i]]]
// CHECK-NEXT:              modelica.bind %[[value]]
// CHECK:               modelica.variable parameter @n : !modelica.int {
// CHECK-NEXT:          ^bb0(%{{.*}}: !modelica.int):
// CHECK-NEXT:              %[[n:.*]] = modelica.constant #modelica<int 4>
// CHECK-NEXT:              modelica.bind %[[n]]
// CHECK:               modelica.variable @x : tensor<4x!modelica.real>
// CHECK:           modelica.variable @y : tensor<3x!modelica.real>
// CHECK:           modelica.equations {
// CHECK-NEXT:          modelica.for_equation %[[i:.*]] : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<3x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK:                       modelica.ref @y
// CHECK:                       %[[rs:.*]] = modelica.ref @rs : tensor<3x!modelica.object<@R>>
// CHECK-NEXT:                  %[[rsi:.*]] = modelica.subscript %[[rs]][%[[i]]]
// CHECK-NEXT:                  modelica.member %[[rsi]], @x : (!modelica.object<@R>) -> tensor<4x!modelica.real>
// CHECK:               modelica.for_equation %[[j:.*]] : !modelica.int in {
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK-NEXT:                  %[[rs:.*]] = modelica.ref @rs
// CHECK-NEXT:                  %[[rsj:.*]] = modelica.subscript %[[rs]][%[[j]]]
// CHECK-NEXT:                  modelica.member %[[rsj]], @x

module {
  modelica.class model @R {
    modelica.component parameter @r : @Real -> !modelica.real
    modelica.component parameter @n : @Integer -> !modelica.int {
      %0 = modelica.constant #modelica<int 2> : !modelica.int
      modelica.bind %0 : !modelica.int
    }
    modelica.component @x : @Real -> tensor<?x!modelica.real> dims [?] {
      %0 = modelica.ref @n : !modelica.int
      modelica.yield %0 : !modelica.int
    }
    modelica.equations {
      modelica.equation {
        %0 = modelica.ref @x : tensor<?x!modelica.real>
        %1 = modelica.ref @r : !modelica.real
        %2 = modelica.ref @n : !modelica.int
        %3 = modelica.fill %1, %2 : (!modelica.real, !modelica.int) -> tensor<?x!modelica.real>
        modelica.equation_sides %0 = %3 : tensor<?x!modelica.real> = tensor<?x!modelica.real>
      }
    }
  }
  modelica.class model @M {
    modelica.component parameter @N : @Integer -> !modelica.int {
      %0 = modelica.constant #modelica<int 3> : !modelica.int
      modelica.bind %0 : !modelica.int
    }
    modelica.component @rs : @R -> tensor<?x!modelica.object<@R>> {
      modelica.modify @r {
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        %1 = modelica.constant #modelica<int 2> : !modelica.int
        %2 = modelica.constant #modelica<int 3> : !modelica.int
        %3 = modelica.array %0, %1, %2 : (!modelica.int, !modelica.int, !modelica.int) -> tensor<3x!modelica.int>
        modelica.bind %3 : tensor<3x!modelica.int>
      }
      modelica.modify each @n {
        %0 = modelica.constant #modelica<int 4> : !modelica.int
        modelica.bind %0 : !modelica.int
      }
    } dims [?] {
      %0 = modelica.ref @N : !modelica.int
      modelica.yield %0 : !modelica.int
    }
    modelica.component @y : @Real -> tensor<?x!modelica.real> dims [?] {
      %0 = modelica.ref @N : !modelica.int
      modelica.yield %0 : !modelica.int
    }
    modelica.equations {
      modelica.for_equation %arg0 : !modelica.int in {
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        %1 = modelica.ref @N : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<?x!modelica.int>
        modelica.yield %2 : tensor<?x!modelica.int>
      } do {
        modelica.equation {
          %0 = modelica.ref @y : tensor<?x!modelica.real>
          %1 = modelica.subscript %0[%arg0] : (tensor<?x!modelica.real>, !modelica.int) -> !modelica.real
          %2 = modelica.ref @rs : tensor<?x!modelica.object<@R>>
          %3 = modelica.subscript %2[%arg0] : (tensor<?x!modelica.object<@R>>, !modelica.int) -> !modelica.object<@R>
          %4 = modelica.member %3, @x : (!modelica.object<@R>) -> tensor<?x!modelica.real>
          %5 = modelica.constant #modelica<int 1> : !modelica.int
          %6 = modelica.subscript %4[%5] : (tensor<?x!modelica.real>, !modelica.int) -> !modelica.real
          modelica.equation_sides %1 = %6 : !modelica.real = !modelica.real
        }
      }
    }
  }
}
