// RUN: modelica-opt %s --split-input-file --convert-model-algorithms | FileCheck %s

// CHECK: bmodelica.model @scalarVariables

bmodelica.model @scalarVariables {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    bmodelica.start @y {
        %0 = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
        bmodelica.yield %0 : !bmodelica.int
    } {each = false, fixed = false}

    // CHECK:       bmodelica.start @y
    // CHECK-NEXT:      %[[start:.*]] = bmodelica.constant #bmodelica<int [[start_value:.*]]> : !bmodelica.int
    // CHECK-NEXT:      bmodelica.yield %[[start]]

    bmodelica.dynamic {
        bmodelica.algorithm {
            %0 = bmodelica.variable_get @x : !bmodelica.int
            bmodelica.variable_set @y, %0 : !bmodelica.int
        }
    }

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [] {
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:       %[[y:.*]] = bmodelica.variable_get @y
    // CHECK-DAG:       %[[res:.*]] = bmodelica.call @[[algorithm_func:.*]](%[[x]])
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[y]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[res]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // CHECK:       bmodelica.dynamic {
    // CHECK-NEXT:      bmodelica.equation_instance %[[t0]]
    // CHECK-NEXT:  }
}

// CHECK:       bmodelica.function @[[algorithm_func]]

// CHECK-DAG:   bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, input>
// CHECK-DAG:   bmodelica.variable @y : !bmodelica.variable<!bmodelica.int, output>

// CHECK:       bmodelica.default @y {
// CHECK-NEXT:      %[[default:.*]] = bmodelica.constant #bmodelica<int 0>
// CHECK-NEXT:      bmodelica.yield %[[default]]
// CHECK-NEXT:  }

// CHECK:       bmodelica.algorithm {
// CHECK-NEXT:      %0 = bmodelica.variable_get @x
// CHECK-NEXT:      bmodelica.variable_set @y, %0
// CHECK-NEXT:  }
