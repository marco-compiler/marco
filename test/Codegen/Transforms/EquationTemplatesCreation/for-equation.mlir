// RUN: modelica-opt %s --split-input-file --create-equation-templates --canonicalize | FileCheck %s

// CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]]] {
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
// CHECK-DAG:       %[[load:.*]] = modelica.load %[[x]][%[[i0]]]
// CHECK-DAG:       %[[zero:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[load]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[zero]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       modelica.main_model {
// CHECK-NEXT:      modelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>}
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<3x!modelica.int>

    modelica.for_equation %i = 0 to 2 {
        modelica.equation {
            %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
            %1 = modelica.load %0[%i] : !modelica.array<3x!modelica.int>
            %2 = modelica.constant #modelica.int<0>
            %3 = modelica.equation_side %1 : tuple<!modelica.int>
            %4 = modelica.equation_side %2 : tuple<!modelica.int>
            modelica.equation_sides %3, %4 : tuple<!modelica.int>, tuple<!modelica.int>
        }
    }
}
