// RUN: marco-opt %s --modelica-instantiate=model=M | FileCheck %s

// Conditional components are removed together with their connections,
// as well as the components and connections
// deselected by selective model extension.
//
// connector Pin
//   Real v;
//   flow Real i;
// end Pin;
// model Component
//   parameter Boolean useAux = false;
//   Pin p;
//   Pin aux if useAux;
// equation
//   connect(p, aux);
// end Component;
// model Base
//   Pin a, b;
//   Component friction;
// equation
//   connect(a, b);
//   connect(a, friction.p);
// end Base;
// model M
//   Component c1;
//   Component c2(useAux = true);
//   extends Base(break friction, break connect(a, b));
// end M;

// CHECK:       modelica.model @M {
// CHECK:           modelica.instance model @c1 : !modelica.object<@Component> {
// CHECK-NOT:           @aux
// CHECK:           modelica.instance model @c2 : !modelica.object<@Component> {
// CHECK:               modelica.instance connector @aux : !modelica.object<@Pin>
// CHECK:           modelica.instance connector @a
// CHECK:           modelica.instance connector @b
// CHECK-NOT:       @friction
// CHECK:           modelica.equations {
// CHECK-NEXT:          modelica.connect {
// CHECK-NEXT:              %[[c2:.*]] = modelica.ref @c2
// CHECK-NEXT:              %[[p:.*]] = modelica.member %[[c2]], @p
// CHECK-NEXT:              %[[c2:.*]] = modelica.ref @c2
// CHECK-NEXT:              %[[aux:.*]] = modelica.member %[[c2]], @aux
// CHECK-NEXT:              modelica.yield %[[p]], %[[aux]]
// CHECK-NEXT:          } {lhsOutside, rhsOutside}
// CHECK-NEXT:      }

module {
  modelica.class connector @Pin {
    modelica.component @v : @Real -> !modelica.real
    modelica.component flow @i : @Real -> !modelica.real
  }
  modelica.class model @Component {
    modelica.component parameter @useAux : @Boolean -> !modelica.bool {
      %0 = modelica.constant #modelica<bool false> : !modelica.bool
      modelica.bind %0 : !modelica.bool
    }
    modelica.component @p : @Pin -> !modelica.object<@Pin>
    modelica.component @aux : @Pin -> !modelica.object<@Pin> if {
      %0 = modelica.ref @useAux : !modelica.bool
      modelica.yield %0 : !modelica.bool
    }
    modelica.equations {
      modelica.connect {
        %0 = modelica.ref @p : !modelica.object<@Pin>
        %1 = modelica.ref @aux : !modelica.object<@Pin>
        modelica.yield %0, %1 : !modelica.object<@Pin>, !modelica.object<@Pin>
      }
    }
  }
  modelica.class model @Base {
    modelica.component @a : @Pin -> !modelica.object<@Pin>
    modelica.component @b : @Pin -> !modelica.object<@Pin>
    modelica.component @friction : @Component -> !modelica.object<@Component>
    modelica.equations {
      modelica.connect {
        %0 = modelica.ref @a : !modelica.object<@Pin>
        %1 = modelica.ref @b : !modelica.object<@Pin>
        modelica.yield %0, %1 : !modelica.object<@Pin>, !modelica.object<@Pin>
      }
      modelica.connect {
        %0 = modelica.ref @a : !modelica.object<@Pin>
        %1 = modelica.ref @friction : !modelica.object<@Component>
        %2 = modelica.member %1, @p : (!modelica.object<@Component>) -> !modelica.object<@Pin>
        modelica.yield %0, %2 : !modelica.object<@Pin>, !modelica.object<@Pin>
      }
    }
  }
  modelica.class model @M {
    modelica.component @c1 : @Component -> !modelica.object<@Component>
    modelica.component @c2 : @Component -> !modelica.object<@Component> {
      modelica.modify @useAux {
        %0 = modelica.constant #modelica<bool true> : !modelica.bool
        modelica.bind %0 : !modelica.bool
      }
    }
    modelica.extends @Base {
      modelica.deselect @friction
      modelica.deselect_connect {
        %0 = modelica.ref @a : !modelica.object<@Pin>
        %1 = modelica.ref @b : !modelica.object<@Pin>
        modelica.yield %0, %1 : !modelica.object<@Pin>, !modelica.object<@Pin>
      }
    }
  }
}
