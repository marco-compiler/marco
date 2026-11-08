// RUN: marco-opt %s --modelica-expand-connections | FileCheck %s

// Connections of connectors of simple types.
//
// connector RealInput = input Real;
// connector RealOutput = output Real;
// model Gain
//   RealInput u;
//   RealOutput y;
// equation
//   y = 2 * u;
// end Gain;
// model Signals
//   Gain g1, g2;
// equation
//   connect(g1.y, g2.u);
// end Signals;

// CHECK-NOT:   modelica.connect
// CHECK:           modelica.equations {
// CHECK:           modelica.equations {
// CHECK-NEXT:          modelica.equation {
// CHECK-NEXT:              %[[g1:.*]] = modelica.ref @g1
// CHECK-NEXT:              %[[y:.*]] = modelica.member %[[g1]], @y
// CHECK-NEXT:              %[[g2:.*]] = modelica.ref @g2
// CHECK-NEXT:              %[[u:.*]] = modelica.member %[[g2]], @u
// CHECK-NEXT:              modelica.equation_sides %[[y]] = %[[u]]

module {
  modelica.model @Signals {
    modelica.instance model @g1 : !modelica.object<@Gain> {
      modelica.variable input @u : !modelica.real
      modelica.variable output @y : !modelica.real
    }
    modelica.instance model @g2 : !modelica.object<@Gain> {
      modelica.variable input @u : !modelica.real
      modelica.variable output @y : !modelica.real
    }
    modelica.equations {
      modelica.connect {
        %0 = modelica.ref @g1 : !modelica.object<@Gain>
        %1 = modelica.member %0, @y : (!modelica.object<@Gain>) -> !modelica.real
        %2 = modelica.ref @g2 : !modelica.object<@Gain>
        %3 = modelica.member %2, @u : (!modelica.object<@Gain>) -> !modelica.real
        modelica.yield %1, %3 : !modelica.real, !modelica.real
      }
      modelica.equation {
        %0 = modelica.ref @g1 : !modelica.object<@Gain>
        %1 = modelica.member %0, @y : (!modelica.object<@Gain>) -> !modelica.real
        %2 = modelica.constant #modelica<int 2> : !modelica.int
        %3 = modelica.ref @g1 : !modelica.object<@Gain>
        %4 = modelica.member %3, @u : (!modelica.object<@Gain>) -> !modelica.real
        %5 = modelica.mul %2, %4 : (!modelica.int, !modelica.real) -> !modelica.real
        modelica.equation_sides %1 = %5 : !modelica.real = !modelica.real
      }
      modelica.equation {
        %0 = modelica.ref @g2 : !modelica.object<@Gain>
        %1 = modelica.member %0, @y : (!modelica.object<@Gain>) -> !modelica.real
        %2 = modelica.constant #modelica<int 2> : !modelica.int
        %3 = modelica.ref @g2 : !modelica.object<@Gain>
        %4 = modelica.member %3, @u : (!modelica.object<@Gain>) -> !modelica.real
        %5 = modelica.mul %2, %4 : (!modelica.int, !modelica.real) -> !modelica.real
        modelica.equation_sides %1 = %5 : !modelica.real = !modelica.real
      }
    }
  }
}
