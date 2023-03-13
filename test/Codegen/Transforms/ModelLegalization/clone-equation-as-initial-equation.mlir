// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(legalize-model{model-name=Test})" | FileCheck %s

// CHECK:       modelica.initial_equation {
// CHECK-NEXT:      %[[x:.*]] = modelica.variable_get @x
// CHECK-NEXT:      %[[rhs_value:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[x]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[rhs_value]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.int>

    modelica.equation {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<0>
        %lhs = modelica.equation_side %0 : tuple<!modelica.int>
        %rhs = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %lhs, %rhs : tuple<!modelica.int>, tuple<!modelica.int>
    }
}
