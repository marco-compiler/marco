// RUN: modelica-opt %s --split-input-file --create-equation-templates --canonicalize | FileCheck %s

// CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [] {
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
// CHECK-DAG:       %[[zero:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[zero]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       modelica.equation_instance %[[t0]]

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>

    modelica.equation {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<0>
        %2 = modelica.equation_side %0 : tuple<!modelica.int>
        %3 = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %2, %3 : tuple<!modelica.int>, tuple<!modelica.int>
    }
}
