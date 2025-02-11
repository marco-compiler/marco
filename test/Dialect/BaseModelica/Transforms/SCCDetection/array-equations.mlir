// RUN: modelica-opt %s --split-input-file --detect-scc --canonicalize-model-for-debug | FileCheck %s

// CHECK-LABEL: @CycleAmongDifferentEquations

bmodelica.model @CycleAmongDifferentEquations {
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<5x!bmodelica.int>

    // COM: x[i] = y[i]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<5x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<5x!bmodelica.int>
        %2 = bmodelica.variable_get @y : tensor<5x!bmodelica.int>
        %3 = bmodelica.tensor_extract %2[%i0] : tensor<5x!bmodelica.int>
        %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // COM: y[i] = 1 - x[i]
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : tensor<5x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<5x!bmodelica.int>
        %2 = bmodelica.constant #bmodelica<int 1>
        %3 = bmodelica.variable_get @x : tensor<5x!bmodelica.int>
        %4 = bmodelica.tensor_extract %3[%i0] : tensor<5x!bmodelica.int>
        %5 = bmodelica.sub %2, %4 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        %6 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.int>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}

    // COM: y[i] = 2 - x[i]
    %t2 = bmodelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @y : tensor<5x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<5x!bmodelica.int>
        %2 = bmodelica.constant #bmodelica<int 2>
        %3 = bmodelica.variable_get @x : tensor<5x!bmodelica.int>
        %4 = bmodelica.tensor_extract %3[%i0] : tensor<5x!bmodelica.int>
        %5 = bmodelica.sub %2, %4 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        %6 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.int>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,4]>, path = #bmodelica<equation_path [L, 0]>}
        bmodelica.matched_equation_instance %t1 {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}
        bmodelica.matched_equation_instance %t2 {indices = #modeling<multidim_range [3,4]>, path = #bmodelica<equation_path [L, 0]>}
    }

    // CHECK:     bmodelica.scc
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {indices = #modeling<multidim_range [0,4]>, path = #bmodelica<equation_path [L, 0]>}
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {indices = #modeling<multidim_range [0,2]>, path = #bmodelica<equation_path [L, 0]>}
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t2]] {indices = #modeling<multidim_range [3,4]>, path = #bmodelica<equation_path [L, 0]>}

    // CHECK-NOT: bmodelica.matched_equation_instance
}

// -----

// CHECK-LABEL: @ArrayBackwardSelfDependency

bmodelica.model @ArrayBackwardSelfDependency {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.real>

    // x[i] = x[i - 1]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<10x!bmodelica.real>
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.sub %i0, %2 : (index, index) -> index
        %4 = bmodelica.tensor_extract %0[%3] : tensor<10x!bmodelica.real>
        %5 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [1,9]>, path = #bmodelica<equation_path [L, 0]>}
    }

    // CHECK:       bmodelica.dynamic
    // CHECK:       bmodelica.scc
    // CHECK-NEXT:  bmodelica.matched_equation_instance %[[t0]]
    // CHECK-SAME:  {
    // CHECK-SAME:      indices = #modeling<multidim_range [1,9]>
    // CHECK-SAME:  }

    // CHECK-NOT:   bmodelica.matched_equation_instance
}

// -----

// CHECK-LABEL: @ArrayForwardSelfDependency

bmodelica.model @ArrayForwardSelfDependency {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.real>

    // x[i] = x[i + 1]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<10x!bmodelica.real>
        %2 = bmodelica.constant 1 : index
        %3 = bmodelica.add %i0, %2 : (index, index) -> index
        %4 = bmodelica.tensor_extract %0[%3] : tensor<10x!bmodelica.real>
        %5 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {indices = #modeling<multidim_range [0,8]>, path = #bmodelica<equation_path [L, 0]>}
    }

    // CHECK:       bmodelica.dynamic
    // CHECK:       bmodelica.scc
    // CHECK-NEXT:  bmodelica.matched_equation_instance %[[t0]]
    // CHECK-SAME:  {
    // CHECK-SAME:      indices = #modeling<multidim_range [0,8]>
    // CHECK-SAME:  }

    // CHECK-NOT:   bmodelica.matched_equation_instance
}
