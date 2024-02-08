// RUN: modelica-opt %s --split-input-file --function-default-values-conversion | FileCheck %s

// CHECK-LABEL: @caller
// CHECK: %[[default:.*]] = modelica.constant #modelica.int<0>
// CHECK: modelica.call @scalarDefaultValue(%[[default]]) : (!modelica.int) -> ()

modelica.function @scalarDefaultValue {
    modelica.variable @x : !modelica.variable<!modelica.int, input>

    modelica.default @x {
        %0 = modelica.constant #modelica.int<0>
        modelica.yield %0 : !modelica.int
    }

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.int
        modelica.print %0 : !modelica.int
    }
}

func.func @caller() {
    modelica.call @scalarDefaultValue() : () -> ()
    func.return
}
