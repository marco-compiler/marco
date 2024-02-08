// RUN: modelica-opt %s --split-input-file --function-default-values-conversion | FileCheck %s

// CHECK-LABEL: @caller
// CHECK-SAME:  (%[[x:.*]]: !modelica.int)
// CHECK-DAG: %[[two:.*]] = modelica.constant #modelica.int<2>
// CHECK-DAG: %[[three:.*]] = modelica.constant #modelica.int<3>
// CHECK-DAG: %[[z:.*]] = modelica.mul %[[x]], %[[two]]
// CHECK-DAG: %[[y:.*]] = modelica.mul %[[z]], %[[three]]
// CHECK: modelica.call @missingArguments(%[[x]], %[[y]], %[[z]]) : (!modelica.int, !modelica.int, !modelica.int) -> ()

modelica.function @missingArguments {
    modelica.variable @x : !modelica.variable<!modelica.int, input>
    modelica.variable @y : !modelica.variable<!modelica.int, input>
    modelica.variable @z : !modelica.variable<!modelica.int, input>

    modelica.default @y {
        %0 = modelica.variable_get @z : !modelica.int
        %1 = modelica.constant #modelica.int<3>
        %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.yield %2 : !modelica.int
    }

    modelica.default @z {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<2>
        %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.yield %2 : !modelica.int
    }

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.variable_get @y : !modelica.int
        %2 = modelica.variable_get @z : !modelica.int
        modelica.print %0 : !modelica.int
        modelica.print %1 : !modelica.int
        modelica.print %2 : !modelica.int
    }
}

func.func @caller(%arg0: !modelica.int) {
    modelica.call @missingArguments(%arg0) : (!modelica.int) -> ()
    func.return
}
