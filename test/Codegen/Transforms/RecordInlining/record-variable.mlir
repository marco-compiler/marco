// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(inline-records{model-name=Test})" | FileCheck %s

// Record with one variable.

// CHECK:       modelica.start @x {
// CHECK-NEXT:      %[[value:.*]] = modelica.constant #modelica.int<0> : !modelica.int
// CHECK-NEXT:      modelica.yield %[[value]]
// CHECK-NEXT:  } {each = true, fixed = false}

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
}

modelica.model @Test {
    modelica.variable @r : !modelica.variable<!modelica.record<@R>>
    modelica.variable @x : !modelica.variable<!modelica.real>

    modelica.equation {
        %0 = modelica.variable_get @r : !modelica.record<@R>
        %1 = modelica.component_get %0, @x : !modelica.record<@R> -> !modelica.real
        %2 = modelica.variable_get @x : !modelica.real
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }
}
