// RUN: modelica-opt %s --split-input-file --mlir-disable-threading --scalar-ranges-equation-split | FileCheck %s

// CHECK-LABEL: @Rank1

bmodelica.model @Rank1 {
    %t0 = bmodelica.equation_template inductions = [%i0] {
        %0 = bmodelica.constant 0.0 : f64
        %1 = bmodelica.equation_side %0 : tuple<f64>
        %2 = bmodelica.equation_side %0 : tuple<f64>
        bmodelica.equation_sides %1, %2 : tuple<f64>, tuple<f64>
    }

    bmodelica.dynamic {
        // CHECK-DAG: bmodelica.equation_instance %{{.*}} indices = {[0,0]}
        // CHECK-DAG: bmodelica.equation_instance %{{.*}} indices = {[2,2]}
        // CHECK-DAG: bmodelica.equation_instance %{{.*}} indices = {[4,4]}
        // CHECK-NOT: bmodelica.equation_instance
        bmodelica.equation_instance %t0, indices = {[0,0],[2,2],[4,4]}
    }
}

// -----

// CHECK-LABEL: @Rank2

bmodelica.model @Rank2 {
    %t0 = bmodelica.equation_template inductions = [%i0, %i1] {
        %0 = bmodelica.constant 0.0 : f64
        %1 = bmodelica.equation_side %0 : tuple<f64>
        %2 = bmodelica.equation_side %0 : tuple<f64>
        bmodelica.equation_sides %1, %2 : tuple<f64>, tuple<f64>
    }

    bmodelica.dynamic {
        // CHECK-DAG: bmodelica.equation_instance %{{.*}} indices = {[0,0][1,1]}
        // CHECK-DAG: bmodelica.equation_instance %{{.*}} indices = {[2,2][2,2]}
        // CHECK-DAG: bmodelica.equation_instance %{{.*}} indices = {[3,3][4,4]}
        // CHECK-NOT: bmodelica.equation_instance
        bmodelica.equation_instance %t0, indices = {[0,0][1,1],[2,2][2,2],[3,3][4,4]}
    }
}
