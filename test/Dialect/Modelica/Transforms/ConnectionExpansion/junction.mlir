// RUN: marco-opt %s --modelica-connection-expansion | FileCheck %s

// Junction of three connectors: the potential variables are equated by two
// equations, and the flow variables are summed by a single equation.
//
// connector Pin
//   Real v;
//   flow Real i;
// end Pin;
// model C
//   Pin p;
// end C;
// model Junction
//   C a, b, c;
// equation
//   connect(a.p, b.p);
//   connect(a.p, c.p);
// end Junction;

// CHECK-NOT:   modelica.connect
// CHECK:           modelica.equations {
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.equations {

// Potential variables, equated to the ones of the representative a.p.
// CHECK-NEXT:          modelica.equation {
// CHECK-NEXT:              %[[b:.*]] = modelica.ref @b
// CHECK-NEXT:              %[[bp:.*]] = modelica.member %[[b]], @p
// CHECK-NEXT:              %[[bv:.*]] = modelica.member %[[bp]], @v
// CHECK-NEXT:              %[[a:.*]] = modelica.ref @a
// CHECK-NEXT:              %[[ap:.*]] = modelica.member %[[a]], @p
// CHECK-NEXT:              %[[av:.*]] = modelica.member %[[ap]], @v
// CHECK-NEXT:              modelica.equation_sides %[[bv]] = %[[av]]
// CHECK-NEXT:          }
// CHECK-NEXT:          modelica.equation {
// CHECK-NEXT:              %[[c:.*]] = modelica.ref @c
// CHECK-NEXT:              %[[cp:.*]] = modelica.member %[[c]], @p
// CHECK-NEXT:              %[[cv:.*]] = modelica.member %[[cp]], @v
// CHECK-NEXT:              %[[a:.*]] = modelica.ref @a
// CHECK-NEXT:              %[[ap:.*]] = modelica.member %[[a]], @p
// CHECK-NEXT:              %[[av:.*]] = modelica.member %[[ap]], @v
// CHECK-NEXT:              modelica.equation_sides %[[cv]] = %[[av]]
// CHECK-NEXT:          }

// Flow variables.
// CHECK-NEXT:          modelica.equation {
// CHECK-NEXT:              %[[a:.*]] = modelica.ref @a
// CHECK-NEXT:              %[[ap:.*]] = modelica.member %[[a]], @p
// CHECK-NEXT:              %[[ai:.*]] = modelica.member %[[ap]], @i
// CHECK-NEXT:              %[[b:.*]] = modelica.ref @b
// CHECK-NEXT:              %[[bp:.*]] = modelica.member %[[b]], @p
// CHECK-NEXT:              %[[bi:.*]] = modelica.member %[[bp]], @i
// CHECK-NEXT:              %[[sum:.*]] = modelica.add %[[ai]], %[[bi]]
// CHECK-NEXT:              %[[c:.*]] = modelica.ref @c
// CHECK-NEXT:              %[[cp:.*]] = modelica.member %[[c]], @p
// CHECK-NEXT:              %[[ci:.*]] = modelica.member %[[cp]], @i
// CHECK-NEXT:              %[[sum2:.*]] = modelica.add %[[sum]], %[[ci]]
// CHECK-NEXT:              %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00>
// CHECK-NEXT:              modelica.equation_sides %[[sum2]] = %[[zero]]
// CHECK-NEXT:          }
// CHECK-NEXT:      }

module {
  modelica.model @Junction {
    modelica.instance @a : !modelica.object<@C> {
      modelica.instance connector @p : !modelica.object<@Pin> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
    }
    modelica.instance @b : !modelica.object<@C> {
      modelica.instance connector @p : !modelica.object<@Pin> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
    }
    modelica.instance @c : !modelica.object<@C> {
      modelica.instance connector @p : !modelica.object<@Pin> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
    }
    modelica.equations {
      modelica.connect {
        %0 = modelica.ref @a : !modelica.object<@C>
        %1 = modelica.member %0, @p : (!modelica.object<@C>) -> !modelica.object<@Pin>
        %2 = modelica.ref @b : !modelica.object<@C>
        %3 = modelica.member %2, @p : (!modelica.object<@C>) -> !modelica.object<@Pin>
        modelica.yield %1, %3 : !modelica.object<@Pin>, !modelica.object<@Pin>
      }
      modelica.connect {
        %0 = modelica.ref @a : !modelica.object<@C>
        %1 = modelica.member %0, @p : (!modelica.object<@C>) -> !modelica.object<@Pin>
        %2 = modelica.ref @c : !modelica.object<@C>
        %3 = modelica.member %2, @p : (!modelica.object<@C>) -> !modelica.object<@Pin>
        modelica.yield %1, %3 : !modelica.object<@Pin>, !modelica.object<@Pin>
      }
    }
  }
}
