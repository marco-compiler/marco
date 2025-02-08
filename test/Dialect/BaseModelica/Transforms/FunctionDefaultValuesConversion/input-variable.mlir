// RUN: modelica-opt %s --split-input-file --function-default-values-conversion | FileCheck %s

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, input>

    bmodelica.default @x {
        %0 = bmodelica.constant #bmodelica<int 0>
        bmodelica.yield %0 : !bmodelica.int
    }

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        bmodelica.print %0 : !bmodelica.int
    }
}

// CHECK-LABEL: @scalar

func.func @scalar() {
    bmodelica.call @foo() : () -> ()

    // CHECK: %[[default:.*]] = bmodelica.constant #bmodelica<int 0>
    // CHECK: bmodelica.call @foo(%[[default]]) : (!bmodelica.int) -> ()

    func.return
}

// -----

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, input>

    bmodelica.default @x {
        %0 = bmodelica.constant #bmodelica.dense_int<[0, 0, 0]> : tensor<3x!bmodelica.int>
        bmodelica.yield %0 : tensor<3x!bmodelica.int>
    }

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : tensor<3x!bmodelica.int>
        bmodelica.print %0 : tensor<3x!bmodelica.int>
    }
}

// CHECK-LABEL: @array

func.func @array() {
    bmodelica.call @foo() : () -> ()

    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica.dense_int<[0, 0, 0]> : tensor<3x!bmodelica.int>
    // CHECK: bmodelica.call @foo(%[[cst]]) : (tensor<3x!bmodelica.int>) -> ()

    func.return
}

// -----

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.int, input>

    bmodelica.default @y {
        %0 = bmodelica.constant #bmodelica<int 1>
        bmodelica.yield %0 : !bmodelica.int
    }

    bmodelica.default @z {
        %0 = bmodelica.constant #bmodelica<int 2>
        bmodelica.yield %0 : !bmodelica.int
    }

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.variable_get @y : !bmodelica.int
        %2 = bmodelica.variable_get @z : !bmodelica.int
        bmodelica.print %0 : !bmodelica.int
        bmodelica.print %1 : !bmodelica.int
        bmodelica.print %2 : !bmodelica.int
    }
}

// CHECK-LABEL: @missingArgsWithDependencies

func.func @missingArgsWithDependencies() {
    %0 = bmodelica.constant #bmodelica<int 0>
    bmodelica.call @foo(%0) : (!bmodelica.int) -> ()

    // CHECK-DAG: %[[x:.*]] = bmodelica.constant #bmodelica<int 0>
    // CHECK-DAG: %[[y:.*]] = bmodelica.constant #bmodelica<int 1>
    // CHECK-DAG: %[[z:.*]] = bmodelica.constant #bmodelica<int 2>
    // CHECK: bmodelica.call @foo(%[[x]], %[[y]], %[[z]]) : (!bmodelica.int, !bmodelica.int, !bmodelica.int) -> ()

    func.return
}

// -----

bmodelica.function @foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.int, input>

    bmodelica.default @y {
        %0 = bmodelica.variable_get @z : !bmodelica.int
        %1 = bmodelica.constant #bmodelica<int 3>
        %2 = bmodelica.mul %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        bmodelica.yield %2 : !bmodelica.int
    }

    bmodelica.default @z {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.constant #bmodelica<int 2>
        %2 = bmodelica.mul %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        bmodelica.yield %2 : !bmodelica.int
    }

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.variable_get @y : !bmodelica.int
        %2 = bmodelica.variable_get @z : !bmodelica.int
        bmodelica.print %0 : !bmodelica.int
        bmodelica.print %1 : !bmodelica.int
        bmodelica.print %2 : !bmodelica.int
    }
}

// CHECK-LABEL: @missingArgsWithDependencies
// CHECK-SAME:  (%[[x:.*]]: !bmodelica.int)

func.func @missingArgsWithDependencies(%arg0: !bmodelica.int) {
    bmodelica.call @foo(%arg0) : (!bmodelica.int) -> ()

    // CHECK-DAG: %[[two:.*]] = bmodelica.constant #bmodelica<int 2>
    // CHECK-DAG: %[[three:.*]] = bmodelica.constant #bmodelica<int 3>
    // CHECK-DAG: %[[z:.*]] = bmodelica.mul %[[x]], %[[two]]
    // CHECK-DAG: %[[y:.*]] = bmodelica.mul %[[z]], %[[three]]
    // CHECK: bmodelica.call @foo(%[[x]], %[[y]], %[[z]]) : (!bmodelica.int, !bmodelica.int, !bmodelica.int) -> ()

    func.return
}
