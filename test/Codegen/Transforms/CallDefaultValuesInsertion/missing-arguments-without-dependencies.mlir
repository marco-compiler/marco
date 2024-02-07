// RUN: modelica-opt %s --split-input-file --call-default-values-insertion | FileCheck %s

// CHECK-LABEL: @caller
// CHECK-DAG: %[[x:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG: %[[y:.*]] = modelica.constant #modelica.int<1>
// CHECK-DAG: %[[z:.*]] = modelica.constant #modelica.int<2>
// CHECK: modelica.call @missingArguments(%[[x]], %[[y]], %[[z]]) : (!modelica.int, !modelica.int, !modelica.int) -> ()

modelica.function @missingArguments {
    modelica.variable @x : !modelica.variable<!modelica.int, input>
    modelica.variable @y : !modelica.variable<!modelica.int, input>
    modelica.variable @z : !modelica.variable<!modelica.int, input>

    modelica.default @y {
        %0 = modelica.constant #modelica.int<1>
        modelica.yield %0 : !modelica.int
    }

    modelica.default @z {
        %0 = modelica.constant #modelica.int<2>
        modelica.yield %0 : !modelica.int
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

func.func @caller() {
    %0 = modelica.constant #modelica.int<0>
    modelica.call @missingArguments(%0) : (!modelica.int) -> ()
    func.return
}
