// RUN: modelica-opt %s --split-input-file --call-default-values-insertion | FileCheck %s

// CHECK-LABEL: @caller
// CHECK: %[[cst:.*]] = modelica.constant #modelica.int_array<[0, 0, 0]> : !modelica.array<3x!modelica.int>
// CHECK: modelica.call @arrayDefaultValue(%[[cst]]) : (!modelica.array<3x!modelica.int>) -> ()

modelica.function @arrayDefaultValue {
    modelica.variable @x : !modelica.variable<3x!modelica.int, input>

    modelica.default @x {
        %0 = modelica.constant #modelica.int_array<[0, 0, 0]> : !modelica.array<3x!modelica.int>
        modelica.yield %0 : !modelica.array<3x!modelica.int>
    }

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        modelica.print %0 : !modelica.array<3x!modelica.int>
    }
}

func.func @caller() {
    modelica.call @arrayDefaultValue() : () -> ()
    func.return
}
