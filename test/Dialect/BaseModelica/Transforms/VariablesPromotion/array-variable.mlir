// RUN: modelica-opt %s --split-input-file --promote-variables-to-parameters --canonicalize | FileCheck %s

// COM: Variable depending on a constant.

// CHECK-LABEL: @constantDependency

bmodelica.model @constantDependency {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>

    // COM: x = 0
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<3x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.int>
        %2 = bmodelica.constant #bmodelica<int 0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0, match = <@x, {[0,2]}> {indices = #modeling<multidim_range [0,2]>}
    }

    // CHECK-NOT: bmodelica.dynamic

    // CHECK:       bmodelica.initial {
    // CHECK-NEXT:      bmodelica.matched_equation_instance %[[t0]]
    // CHECK-SAME:      match = <@x, {[0,2]}>
    // CHECK-SAME:      {
    // CHECK-SAME:          indices = #modeling<multidim_range [0,2]>
    // CHECK-SAME:      }
    // CHECK-NEXT:  }

    // CHECK-NOT: bmodelica.matched_equation_instance
}

// -----

// COM: Variable depending on a parameter.

// CHECK-LABEL: @parameterDependency

bmodelica.model @parameterDependency {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>
    // CHECK-DAG: bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int, parameter>

    // COM: x[i] = 0
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<3x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.int>
        %2 = bmodelica.constant #bmodelica<int 0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // COM: y[i] = x[i]
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : tensor<3x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.int>
        %2 = bmodelica.variable_get @x : tensor<3x!bmodelica.int>
        %3 = bmodelica.tensor_extract %2[%i0] : tensor<3x!bmodelica.int>
        %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}

    bmodelica.initial {
        bmodelica.matched_equation_instance %t0, match = <@x, {[0,2]}> {indices = #modeling<multidim_range [0,2]>}
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t1, match = <@y, {[0,2]}> {indices = #modeling<multidim_range [0,2]>}
    }

    // CHECK-NOT: bmodelica.dynamic

    // CHECK:     bmodelica.initial
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]], match = <@x, {[0,2]}>
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]], match = <@y, {[0,2]}>

    // CHECK-NOT: bmodelica.matched_equation_instance
}

// -----

// COM: Variable depending on the time variable.

// CHECK-LABEL: @timeDependency

bmodelica.model @timeDependency {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>

    // COM: x[i] = time
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<3x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.int>
        %2 = bmodelica.time : !bmodelica.real
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.int>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0, match = <@x, {[0,2]}> {indices = #modeling<multidim_range [0,2]>}
    }

    // CHECk-NOT: bmodelica.initial

    // CHECK: bmodelica.dynamic
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]], match = <@x, {[0,2]}>

    // CHECK-NOT: bmodelica.matched_equation_instance
}

// -----

// COM: Variable z depending on the non-parameter variable y.
// COM: Variable y depending on the parameter x.

// CHECK-LABEL: @parameterDependencyPropagation

bmodelica.model @parameterDependencyPropagation {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int>
    bmodelica.variable @z : !bmodelica.variable<3x!bmodelica.int>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>
    // CHECK-DAG: bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int, parameter>
    // CHECK-DAG: bmodelica.variable @z : !bmodelica.variable<3x!bmodelica.int, parameter>

    // COM: x[i] = 0
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<3x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.int>
        %2 = bmodelica.constant #bmodelica<int 0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // COM: y[i] = x[i]
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : tensor<3x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.int>
        %2 = bmodelica.variable_get @x : tensor<3x!bmodelica.int>
        %3 = bmodelica.tensor_extract %2[%i0] : tensor<3x!bmodelica.int>
        %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}

    // COM: z[i] = y[i]
    %t2 = bmodelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @z : tensor<3x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.int>
        %2 = bmodelica.variable_get @y : tensor<3x!bmodelica.int>
        %3 = bmodelica.tensor_extract %2[%i0] : tensor<3x!bmodelica.int>
        %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}

    bmodelica.initial {
        bmodelica.matched_equation_instance %t0, match = <@x, {[0,2]}> {indices = #modeling<multidim_range [0,2]>}
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t1, match = <@y, {[0,2]}> {indices = #modeling<multidim_range [0,2]>}
        bmodelica.matched_equation_instance %t2, match = <@z, {[0,2]}> {indices = #modeling<multidim_range [0,2]>}
    }

    // CHECK-NOT: bmodelica.dynamic

    // CHECK: bmodelica.initial
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]], match = <@x, {[0,2]}>
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]], match = <@y, {[0,2]}>
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t2]], match = <@z, {[0,2]}>

    // CHECK-NOT: bmodelica.matched_equation_instance
}

// -----

// COM: Promotable SCC.

// CHECK-LABEL: @promotableSCC

bmodelica.model @promotableSCC {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>
    // CHECK-DAG: bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int, parameter>

    // COM: x[i] = y[i]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<3x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.int>
        %2 = bmodelica.variable_get @y : tensor<3x!bmodelica.int>
        %3 = bmodelica.tensor_extract %2[%i0] : tensor<3x!bmodelica.int>
        %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // COM: y[i] = x[i]
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : tensor<3x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.int>
        %2 = bmodelica.variable_get @x : tensor<3x!bmodelica.int>
        %3 = bmodelica.tensor_extract %2[%i0] : tensor<3x!bmodelica.int>
        %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0, match = <@x, {[0,2]}> {indices = #modeling<multidim_range [0,2]>}
        bmodelica.matched_equation_instance %t1, match = <@y, {[0,2]}> {indices = #modeling<multidim_range [0,2]>}
    }

    // CHECK-NOT: bmodelica.dynamic

    // CHECK: bmodelica.initial
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]], match = <@x, {[0,2]}>
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]], match = <@y, {[0,2]}>

    // CHECK-NOT: bmodelica.matched_equation_instance
}

// -----

// COM: Promotable SCC depending on a promotable variable.

// CHECK-LABEL: @promotableSCCDependingOnPromotableVar

bmodelica.model @promotableSCCDependingOnPromotableVar {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int>
    bmodelica.variable @z : !bmodelica.variable<3x!bmodelica.int>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>
    // CHECK-DAG: bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int, parameter>
    // CHECK-DAG: bmodelica.variable @z : !bmodelica.variable<3x!bmodelica.int, parameter>

    // COM: x[i] = 0
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<3x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.int>
        %2 = bmodelica.constant #bmodelica<int 0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // COM: y[i] = x[i] + z[i]
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : tensor<3x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.int>
        %2 = bmodelica.variable_get @x : tensor<3x!bmodelica.int>
        %3 = bmodelica.tensor_extract %2[%i0] : tensor<3x!bmodelica.int>
        %4 = bmodelica.variable_get @z : tensor<3x!bmodelica.int>
        %5 = bmodelica.tensor_extract %4[%i0] : tensor<3x!bmodelica.int>
        %6 = bmodelica.add %3, %5 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        %7 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.int>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}

    // COM: z[i] = x[i] + y[i]
    %t2 = bmodelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @z : tensor<3x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.int>
        %2 = bmodelica.variable_get @x : tensor<3x!bmodelica.int>
        %3 = bmodelica.tensor_extract %2[%i0] : tensor<3x!bmodelica.int>
        %4 = bmodelica.variable_get @y : tensor<3x!bmodelica.int>
        %5 = bmodelica.tensor_extract %4[%i0] : tensor<3x!bmodelica.int>
        %6 = bmodelica.add %3, %5 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        %7 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.int>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0, match = <@x, {[0,2]}> {indices = #modeling<multidim_range [0,2]>}
        bmodelica.matched_equation_instance %t1, match = <@y, {[0,2]}> {indices = #modeling<multidim_range [0,2]>}
        bmodelica.matched_equation_instance %t2, match = <@z, {[0,2]}> {indices = #modeling<multidim_range [0,2]>}
    }

    // CHECK-NOT: bmodelica.dynamic

    // CHECK: bmodelica.initial
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]], match = <@x, {[0,2]}>
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]], match = <@y, {[0,2]}>
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t2]], match = <@z, {[0,2]}>

    // CHECK-NOT: bmodelica.matched_equation_instance
}

// -----

// COM: Variable depending on a variable that is not written by any other equation
// COM: (and, thus, potentially a state variable).

// CHECK-LABEL: @varDependingOnUnwrittenVar

bmodelica.model @varDependingOnUnwrittenVar {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>
    // CHECK-DAG: bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.int>

    // COM: y[i] = x[i]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @y : tensor<3x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<3x!bmodelica.int>
        %2 = bmodelica.variable_get @x : tensor<3x!bmodelica.int>
        %3 = bmodelica.tensor_extract %2[%i0] : tensor<3x!bmodelica.int>
        %4 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0, match = <@y, {[0,2]}> {indices = #modeling<multidim_range [0,2]>}
    }

    // CHECK-NOT: bmodelica.initial

    // CHECK: bmodelica.dynamic
    // CHECK: bmodelica.matched_equation_instance %[[t0]], match = <@y, {[0,2]}>

    // CHECK-NOT: bmodelica.matched_equation_instance
}

// -----

// COM: Promotable array written by multiple equations.

// CHECK-LABEL: @promotableVarWrittenByMultipleEquations

bmodelica.model @promotableVarWrittenByMultipleEquations {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real, parameter>

    // COM: x[i] = 1
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<2x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 0.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // COM: x[i] = 1
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<2x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 1.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0, match = <@x, {[0,0]}> {indices = #modeling<multidim_range [0,0]>}
        bmodelica.matched_equation_instance %t1, match = <@x, {[1,1]}> {indices = #modeling<multidim_range [1,1]>}
    }

    // CHECK-NOT: bmodelica.dynamic

    // CHECK: bmodelica.initial
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]], match = <@x, {[0,0]}> {indices = #modeling<multidim_range [0,0]>}
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]], match = <@x, {[1,1]}> {indices = #modeling<multidim_range [1,1]>}

    // CHECK-NOT: bmodelica.matched_equation_instance
}

// -----

// COM: Array not fully promotable.

// CHECK-LABEL: @notFullyPromotableVar

bmodelica.model @notFullyPromotableVar {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>

    // COM: x[i] = 0
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<2x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 0.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // COM: x[i] = time
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<2x!bmodelica.real>
        %2 = bmodelica.time : !bmodelica.real
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0, match = <@x, {[0,0]}> {indices = #modeling<multidim_range [0,0]>}
        bmodelica.matched_equation_instance %t1, match = <@x, {[1,1]}> {indices = #modeling<multidim_range [1,1]>}
    }

    // CHECK-NOT: bmodelica.initial

    // CHECK: bmodelica.dynamic
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]], match = <@x, {[0,0]}>
    // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]], match = <@x, {[1,1]}>

    // CHECK-NOT: bmodelica.matched_equation_instance
}
