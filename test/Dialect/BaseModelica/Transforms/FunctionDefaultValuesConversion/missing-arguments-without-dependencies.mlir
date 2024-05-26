// RUN: modelica-opt %s --split-input-file --function-default-values-conversion | FileCheck %s

// CHECK-LABEL: @caller
// CHECK-DAG: %[[x:.*]] = bmodelica.constant #bmodelica<int 0>
// CHECK-DAG: %[[y:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK-DAG: %[[z:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK: bmodelica.call @missingArguments(%[[x]], %[[y]], %[[z]]) : (!bmodelica.int, !bmodelica.int, !bmodelica.int) -> ()

bmodelica.function @missingArguments {
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

func.func @caller() {
    %0 = bmodelica.constant #bmodelica<int 0>
    bmodelica.call @missingArguments(%0) : (!bmodelica.int) -> ()
    func.return
}
