// RUN: modelica-opt %s --split-input-file --single-valued-induction-elimination | FileCheck %s

// CHECK:   %[[t0:.*]] = modelica.equation_template inductions = [] {
// CHECK:       %[[i0:.*]] = modelica.constant 0 : index
// CHECK:       modelica.load %{{.*}}[%[[i0:.*]]]

// CHECK:   %[[t1:.*]] = modelica.equation_template inductions = [] {
// CHECK:       %[[i0:.*]] = modelica.constant 1 : index
// CHECK:       modelica.load %{{.*}}[%[[i0:.*]]]

// CHECK:       modelica.main_model {
// CHECK-DAG:       modelica.matched_equation_instance %[[t0]] {path = #modelica<equation_path [L, 0]>}
// CHECK-DAG:       modelica.matched_equation_instance %[[t1]] {path = #modelica<equation_path [L, 0]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<2x!modelica.real>

    %t0 = modelica.equation_template inductions = [%i0] {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
        %1 = modelica.load %0[%i0] : !modelica.array<2x!modelica.real>
        %2 = modelica.constant #modelica.real<0.0>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,0]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
        modelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [1,1]>, path = #modelica<equation_path [L, 0]>} : !modelica.equation
    }
}
