// RUN: modelica-opt %s --split-input-file --mlir-disable-threading --scalar-ranges-equation-split | FileCheck %s

// CHECK-LABEL: @Test

bmodelica.model @Test {
    %t0 = bmodelica.equation_template inductions = [%i0] {
        %0 = bmodelica.constant 0.0 : f64
        %1 = bmodelica.equation_side %0 : tuple<f64>
        %2 = bmodelica.equation_side %0 : tuple<f64>
        bmodelica.equation_sides %1, %2 : tuple<f64>, tuple<f64>
    }

    bmodelica.dynamic {
        // CHECK-DAG: bmodelica.equation_instance %{{.*}} indices = {[0,1],[3,4]}
        // CHECK-NOT: bmodelica.equation_instance
        bmodelica.equation_instance %t0, indices = {[0,1],[3,4]}
    }
}
