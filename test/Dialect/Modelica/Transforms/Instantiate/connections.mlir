// RUN: marco-opt %s --modelica-instantiate=model=M | FileCheck %s

// Connect-equations are kept, and the connectors are marked as outside if
// they are connectors of the instance whose class contains the connection.
// The ranges and conditions of the for- and
// if-equations containing them are evaluated, unrolling
// the loops whose body contains conditional connections, and the parameters
// used as indices are replaced by their values.
//
// connector Pin
//   Real v;
//   flow Real i;
// end Pin;
// model Inner
//   Pin p, n;
// equation
//   connect(p, n);
// end Inner;
// model M
//   parameter Integer m = 3;
//   Inner c[m];
//   Pin pins[m];
// equation
//   for i in 1:m loop
//     if i > 1 then
//       connect(pins[i - 1], pins[i]);
//     end if;
//   end for;
//   for i in 1:m - 1 loop
//     connect(c[i].n, c[i + 1].p);
//   end for;
//   connect(pins[m], c[m].n);
// end M;

// CHECK:       modelica.model @M {
// CHECK:           modelica.equations {

// Unrolled conditional connections.
// CHECK-NEXT:          modelica.connect {
// CHECK-DAG:               %[[one:.*]] = modelica.constant #modelica<int 1>
// CHECK-DAG:               %[[two:.*]] = modelica.constant #modelica<int 2>
// CHECK-DAG:               %[[lhs:.*]] = modelica.subscript %{{.*}}[%[[one]]]
// CHECK-DAG:               %[[rhs:.*]] = modelica.subscript %{{.*}}[%[[two]]]
// CHECK:                   modelica.yield %[[lhs]], %[[rhs]]
// CHECK-NEXT:          } {lhsOutside, rhsOutside}
// CHECK-NEXT:          modelica.connect {
// CHECK:               } {lhsOutside, rhsOutside}

// Connections between inside connectors.
// CHECK-NEXT:          modelica.for_equation %[[i:.*]] : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<2x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.connect {
// CHECK:                   }{{$}}

// Parameters used as indices.
// CHECK:               modelica.connect {
// CHECK-NEXT:              %[[pins:.*]] = modelica.ref @pins
// CHECK-NEXT:              %[[m:.*]] = modelica.constant #modelica<int 3>
// CHECK-NEXT:              modelica.subscript %[[pins]][%[[m]]]
// CHECK:               } {lhsOutside}

// Connections within the elements of arrays of instances.
// CHECK:               modelica.for_equation
// CHECK:                   modelica.connect {
// CHECK:                   } {lhsOutside, rhsOutside}

module {
  modelica.class connector @Pin {
    modelica.component @v : @Real -> !modelica.real
    modelica.component flow @i : @Real -> !modelica.real
  }
  modelica.class model @Inner {
    modelica.component @p : @Pin -> !modelica.object<@Pin>
    modelica.component @n : @Pin -> !modelica.object<@Pin>
    modelica.equations {
      modelica.connect {
        %0 = modelica.ref @p : !modelica.object<@Pin>
        %1 = modelica.ref @n : !modelica.object<@Pin>
        modelica.yield %0, %1 : !modelica.object<@Pin>, !modelica.object<@Pin>
      }
    }
  }
  modelica.class model @M {
    modelica.component parameter @m : @Integer -> !modelica.int {
      %0 = modelica.constant #modelica<int 3> : !modelica.int
      modelica.bind %0 : !modelica.int
    }
    modelica.component @c : @Inner -> tensor<?x!modelica.object<@Inner>> dims [?] {
      %0 = modelica.ref @m : !modelica.int
      modelica.yield %0 : !modelica.int
    }
    modelica.component @pins : @Pin -> tensor<?x!modelica.object<@Pin>> dims [?] {
      %0 = modelica.ref @m : !modelica.int
      modelica.yield %0 : !modelica.int
    }
    modelica.equations {
      modelica.for_equation %arg0 : !modelica.int in {
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        %1 = modelica.ref @m : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<?x!modelica.int>
        modelica.yield %2 : tensor<?x!modelica.int>
      } do {
        modelica.if_equation {
          %0 = modelica.constant #modelica<int 1> : !modelica.int
          %1 = modelica.gt %arg0, %0 : (!modelica.int, !modelica.int) -> !modelica.bool
          modelica.yield %1 : !modelica.bool
        } then {
          modelica.connect {
            %0 = modelica.ref @pins : tensor<?x!modelica.object<@Pin>>
            %1 = modelica.constant #modelica<int 1> : !modelica.int
            %2 = modelica.sub %arg0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
            %3 = modelica.subscript %0[%2] : (tensor<?x!modelica.object<@Pin>>, !modelica.int) -> !modelica.object<@Pin>
            %4 = modelica.ref @pins : tensor<?x!modelica.object<@Pin>>
            %5 = modelica.subscript %4[%arg0] : (tensor<?x!modelica.object<@Pin>>, !modelica.int) -> !modelica.object<@Pin>
            modelica.yield %3, %5 : !modelica.object<@Pin>, !modelica.object<@Pin>
          }
        }
      }
      modelica.for_equation %arg0 : !modelica.int in {
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        %1 = modelica.ref @m : !modelica.int
        %2 = modelica.constant #modelica<int 1> : !modelica.int
        %3 = modelica.sub %1, %2 : (!modelica.int, !modelica.int) -> !modelica.int
        %4 = modelica.range %0 to %3 : (!modelica.int, !modelica.int) -> tensor<?x!modelica.int>
        modelica.yield %4 : tensor<?x!modelica.int>
      } do {
        modelica.connect {
          %0 = modelica.ref @c : tensor<?x!modelica.object<@Inner>>
          %1 = modelica.subscript %0[%arg0] : (tensor<?x!modelica.object<@Inner>>, !modelica.int) -> !modelica.object<@Inner>
          %2 = modelica.member %1, @n : (!modelica.object<@Inner>) -> !modelica.object<@Pin>
          %3 = modelica.ref @c : tensor<?x!modelica.object<@Inner>>
          %4 = modelica.constant #modelica<int 1> : !modelica.int
          %5 = modelica.add %arg0, %4 : (!modelica.int, !modelica.int) -> !modelica.int
          %6 = modelica.subscript %3[%5] : (tensor<?x!modelica.object<@Inner>>, !modelica.int) -> !modelica.object<@Inner>
          %7 = modelica.member %6, @p : (!modelica.object<@Inner>) -> !modelica.object<@Pin>
          modelica.yield %2, %7 : !modelica.object<@Pin>, !modelica.object<@Pin>
        }
      }
      modelica.connect {
        %0 = modelica.ref @pins : tensor<?x!modelica.object<@Pin>>
        %1 = modelica.ref @m : !modelica.int
        %2 = modelica.subscript %0[%1] : (tensor<?x!modelica.object<@Pin>>, !modelica.int) -> !modelica.object<@Pin>
        %3 = modelica.ref @c : tensor<?x!modelica.object<@Inner>>
        %4 = modelica.ref @m : !modelica.int
        %5 = modelica.subscript %3[%4] : (tensor<?x!modelica.object<@Inner>>, !modelica.int) -> !modelica.object<@Inner>
        %6 = modelica.member %5, @n : (!modelica.object<@Inner>) -> !modelica.object<@Pin>
        modelica.yield %2, %6 : !modelica.object<@Pin>, !modelica.object<@Pin>
      }
    }
  }
}
