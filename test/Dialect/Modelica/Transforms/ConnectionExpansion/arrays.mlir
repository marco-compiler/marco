// RUN: marco-opt %s --modelica-connection-expansion | FileCheck %s

// Connections of whole arrays of connectors and of slices of them, whose
// connectors contain array variables. Each selected dimension of the
// connectors and each dimension of the variables becomes a dimension of the
// equations, the indices being zero-based.
//
// connector Pin
//   Real v[2];
//   flow Real i[2];
// end Pin;
// model C
//   Pin p;
// end C;
// model Arrays
//   C a[3], b[4];
// equation
//   connect(a.p, b[2:4].p);
// end Arrays;

// CHECK-NOT:   modelica.connect
// CHECK:           modelica.equations {
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.equations {

// Potential variables.
// CHECK-NEXT:          modelica.for_equation %[[k:.*]] : !modelica.int in {
// CHECK-NEXT:              %[[from:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:              %[[to:.*]] = modelica.constant #modelica<int 3>
// CHECK-NEXT:              %[[range:.*]] = modelica.range %[[from]] to %[[to]] {{.*}} -> tensor<3x!modelica.int>
// CHECK-NEXT:              modelica.yield %[[range]]
// CHECK-NEXT:          } do {
// CHECK-NEXT:              modelica.for_equation %[[j:.*]] : !modelica.int in {
// CHECK-NEXT:                  %[[from:.*]] = modelica.constant #modelica<int 0>
// CHECK-NEXT:                  %[[to:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:                  %[[range:.*]] = modelica.range %[[from]] to %[[to]] {{.*}} -> tensor<2x!modelica.int>
// CHECK-NEXT:                  modelica.yield %[[range]]
// CHECK-NEXT:              } do {
// CHECK-NEXT:                  modelica.equation {
// CHECK-NEXT:                      %[[b:.*]] = modelica.ref @b
// CHECK-NEXT:                      %[[bk:.*]] = modelica.subscript %[[b]][%[[k]]]
// CHECK-NEXT:                      %[[bp:.*]] = modelica.member %[[bk]], @p
// CHECK-NEXT:                      %[[bv:.*]] = modelica.member %[[bp]], @v
// CHECK-NEXT:                      %[[bvj:.*]] = modelica.subscript %[[bv]][%[[j]]]
// CHECK-NEXT:                      %[[minusOne:.*]] = modelica.constant #modelica<int -1>
// CHECK-NEXT:                      %[[previous:.*]] = modelica.add %[[k]], %[[minusOne]]
// CHECK-NEXT:                      %[[a:.*]] = modelica.ref @a
// CHECK-NEXT:                      %[[ak:.*]] = modelica.subscript %[[a]][%[[previous]]]
// CHECK-NEXT:                      %[[ap:.*]] = modelica.member %[[ak]], @p
// CHECK-NEXT:                      %[[av:.*]] = modelica.member %[[ap]], @v
// CHECK-NEXT:                      %[[avj:.*]] = modelica.subscript %[[av]][%[[j]]]
// CHECK-NEXT:                      modelica.equation_sides %[[bvj]] = %[[avj]]
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }

// Flow variables.
// CHECK-NEXT:          modelica.for_equation %[[k:.*]] : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<3x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.for_equation %[[j:.*]] : !modelica.int in {
// CHECK:                       modelica.range {{.*}} -> tensor<2x!modelica.int>
// CHECK:                   } do {
// CHECK-NEXT:                  modelica.equation {
// CHECK-NEXT:                      %[[a:.*]] = modelica.ref @a
// CHECK-NEXT:                      %[[ak:.*]] = modelica.subscript %[[a]][%[[k]]]
// CHECK-NEXT:                      %[[ap:.*]] = modelica.member %[[ak]], @p
// CHECK-NEXT:                      %[[ai:.*]] = modelica.member %[[ap]], @i
// CHECK-NEXT:                      %[[aij:.*]] = modelica.subscript %[[ai]][%[[j]]]
// CHECK-NEXT:                      %[[one:.*]] = modelica.constant #modelica<int 1>
// CHECK-NEXT:                      %[[next:.*]] = modelica.add %[[k]], %[[one]]
// CHECK-NEXT:                      %[[b:.*]] = modelica.ref @b
// CHECK-NEXT:                      %[[bk:.*]] = modelica.subscript %[[b]][%[[next]]]
// CHECK-NEXT:                      %[[bp:.*]] = modelica.member %[[bk]], @p
// CHECK-NEXT:                      %[[bi:.*]] = modelica.member %[[bp]], @i
// CHECK-NEXT:                      %[[bij:.*]] = modelica.subscript %[[bi]][%[[j]]]
// CHECK-NEXT:                      %[[sum:.*]] = modelica.add %[[aij]], %[[bij]]
// CHECK-NEXT:                      %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:                      modelica.equation_sides %[[sum]] = %[[zero]]
// CHECK-NEXT:                  }
// CHECK-NEXT:              }
// CHECK-NEXT:          }

// Unconnected flow variables of b[1].
// CHECK-NEXT:          modelica.for_equation %[[j:.*]] : !modelica.int in {
// CHECK:                   modelica.range {{.*}} -> tensor<2x!modelica.int>
// CHECK:               } do {
// CHECK-NEXT:              modelica.equation {
// CHECK-NEXT:                  %[[first:.*]] = modelica.constant #modelica<int 0>
// CHECK-NEXT:                  %[[b:.*]] = modelica.ref @b
// CHECK-NEXT:                  %[[bk:.*]] = modelica.subscript %[[b]][%[[first]]]
// CHECK-NEXT:                  %[[bp:.*]] = modelica.member %[[bk]], @p
// CHECK-NEXT:                  %[[bi:.*]] = modelica.member %[[bp]], @i
// CHECK-NEXT:                  %[[bij:.*]] = modelica.subscript %[[bi]][%[[j]]]
// CHECK-NEXT:                  %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:                  modelica.equation_sides %[[bij]] = %[[zero]]
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      }

module {
  modelica.model @Arrays {
    modelica.instance @a : tensor<3x!modelica.object<@C>> {
      modelica.instance connector @p : !modelica.object<@Pin> {
        modelica.variable @v : tensor<2x!modelica.real>
        modelica.variable flow @i : tensor<2x!modelica.real>
      }
    }
    modelica.instance @b : tensor<4x!modelica.object<@C>> {
      modelica.instance connector @p : !modelica.object<@Pin> {
        modelica.variable @v : tensor<2x!modelica.real>
        modelica.variable flow @i : tensor<2x!modelica.real>
      }
    }
    modelica.equations {
      modelica.connect {
        %0 = modelica.ref @a : tensor<3x!modelica.object<@C>>
        %1 = modelica.member %0, @p : (tensor<3x!modelica.object<@C>>) -> tensor<3x!modelica.object<@Pin>>
        %2 = modelica.ref @b : tensor<4x!modelica.object<@C>>
        %3 = modelica.constant #modelica<int 1> : !modelica.int
        %4 = modelica.constant #modelica<int 3> : !modelica.int
        %5 = modelica.range %3 to %4 : (!modelica.int, !modelica.int) -> tensor<3x!modelica.int>
        %6 = modelica.subscript %2[%5] : (tensor<4x!modelica.object<@C>>, tensor<3x!modelica.int>) -> tensor<3x!modelica.object<@C>>
        %7 = modelica.member %6, @p : (tensor<3x!modelica.object<@C>>) -> tensor<3x!modelica.object<@Pin>>
        modelica.yield %1, %7 : tensor<3x!modelica.object<@Pin>>, tensor<3x!modelica.object<@Pin>>
      }
    }
  }
}
