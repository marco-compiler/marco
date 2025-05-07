// RUN: modelica-opt %s --split-input-file --mlir-disable-threading --scalar-ranges-equation-split | FileCheck %s

// CHECK-LABEL: @ScalarRanges

bmodelica.model @ScalarRanges {
    bmodelica.variable @x : !bmodelica.variable<10xf64>
    
    %t0 = bmodelica.equation_template inductions = [%i0] {
        %0 = bmodelica.variable_get @x : tensor<10xf64>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<10xf64>
        %2 = bmodelica.constant 0.0 : f64
        %3 = bmodelica.equation_side %1 : tuple<f64>
        %4 = bmodelica.equation_side %2 : tuple<f64>
        bmodelica.equation_sides %3, %4 : tuple<f64>, tuple<f64>
    }

    bmodelica.dynamic {
        // CHECK-DAG: bmodelica.equation_instance %{{.*}} indices = {[0,0]}, match = <@x, {[0,0]}>
        // CHECK-DAG: bmodelica.equation_instance %{{.*}} indices = {[2,2]}, match = <@x, {[2,2]}>
        // CHECK-DAG: bmodelica.equation_instance %{{.*}} indices = {[4,4]}, match = <@x, {[4,4]}>
        // CHECK-NOT: bmodelica.equation_instance
        bmodelica.equation_instance %t0, indices = {[0,0],[2,2],[4,4]}, match = <@x, {[0,0],[2,2],[4,4]}>
    }
}

// -----

// CHECK-LABEL: @NonScalarRanges

bmodelica.model @NonScalarRanges {
    bmodelica.variable @x : !bmodelica.variable<10xf64>

    %t0 = bmodelica.equation_template inductions = [%i0] {
        %0 = bmodelica.variable_get @x : tensor<10xf64>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<10xf64>
        %2 = bmodelica.constant 0.0 : f64
        %3 = bmodelica.equation_side %1 : tuple<f64>
        %4 = bmodelica.equation_side %2 : tuple<f64>
        bmodelica.equation_sides %3, %4 : tuple<f64>, tuple<f64>
    }

    bmodelica.dynamic {
        // CHECK-DAG: bmodelica.equation_instance %{{.*}} indices = {[0,1],[3,4]}, match = <@x, {[0,1],[3,4]}>
        // CHECK-NOT: bmodelica.equation_instance
        bmodelica.equation_instance %t0, indices = {[0,1],[3,4]}, match = <@x, {[0,1],[3,4]}>
    }
}

// -----

// CHECK-LABEL: @MixedScalarAndNonScalarRanges

bmodelica.model @MixedScalarAndNonScalarRanges {
    bmodelica.variable @x : !bmodelica.variable<10xf64>

    %t0 = bmodelica.equation_template inductions = [%i0] {
        %0 = bmodelica.variable_get @x : tensor<10xf64>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<10xf64>
        %2 = bmodelica.constant 0.0 : f64
        %3 = bmodelica.equation_side %1 : tuple<f64>
        %4 = bmodelica.equation_side %2 : tuple<f64>
        bmodelica.equation_sides %3, %4 : tuple<f64>, tuple<f64>
    }

    bmodelica.dynamic {
        // CHECK-DAG: bmodelica.equation_instance %{{.*}} indices = {[0,0]}, match = <@x, {[0,0]}>
        // CHECK-DAG: bmodelica.equation_instance %{{.*}} indices = {[2,2]}, match = <@x, {[2,2]}>
        // CHECK-DAG: bmodelica.equation_instance %{{.*}} indices = {[4,5],[7,8]}, match = <@x, {[4,5],[7,8]}>
        // CHECK-NOT: bmodelica.equation_instance
        bmodelica.equation_instance %t0, indices = {[0,0],[2,2],[4,5],[7,8]}, match = <@x, {[0,0],[2,2],[4,5],[7,8]}>
    }
}
