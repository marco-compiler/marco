// RUN: modelica-opt %s --split-input-file --function-default-values-conversion | FileCheck %s

// CHECK-LABEL: @caller
// CHECK-SAME:  (%[[x:.*]]: !bmodelica.int)
// CHECK-DAG: %[[two:.*]] = bmodelica.constant #bmodelica.int<2>
// CHECK-DAG: %[[three:.*]] = bmodelica.constant #bmodelica.int<3>
// CHECK-DAG: %[[z:.*]] = bmodelica.mul %[[x]], %[[two]]
// CHECK-DAG: %[[y:.*]] = bmodelica.mul %[[z]], %[[three]]
// CHECK: bmodelica.call @missingArguments(%[[x]], %[[y]], %[[z]]) : (!bmodelica.int, !bmodelica.int, !bmodelica.int) -> ()

bmodelica.function @missingArguments {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.int, input>

    bmodelica.default @y {
        %0 = bmodelica.variable_get @z : !bmodelica.int
        %1 = bmodelica.constant #bmodelica.int<3>
        %2 = bmodelica.mul %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        bmodelica.yield %2 : !bmodelica.int
    }

    bmodelica.default @z {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.constant #bmodelica.int<2>
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

func.func @caller(%arg0: !bmodelica.int) {
    bmodelica.call @missingArguments(%arg0) : (!bmodelica.int) -> ()
    func.return
}
