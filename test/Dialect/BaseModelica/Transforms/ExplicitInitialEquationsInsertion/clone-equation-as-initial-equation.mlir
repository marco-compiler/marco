// RUN: modelica-opt %s --split-input-file --insert-explicit-initial-equations | FileCheck %s

// CHECK-LABEL: @scalarVariable

bmodelica.model @scalarVariable {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    %t0 = bmodelica.equation_template inductions = [] {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.constant #bmodelica<int 0>
        %lhs = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %rhs = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0
    }

    // CHECK: %[[t0:.*]] = bmodelica.equation_template

    // CHECK:       bmodelica.initial
    // CHECK-NEXT:  bmodelica.equation_instance %[[t0]]

    // CHECK:       bmodelica.dynamic
    // CHECK-NEXT:  bmodelica.equation_instance %[[t0]]
}
