// RUN: modelica-opt %s --auto-diff | FileCheck %s

modelica.function @foo(%arg0 : !modelica.ptr<2x!modelica.real>, %arg1 : !modelica.int, %arg2 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y", "z"], results_names = ["t"], derivative = #modelica.derivative<"foo1", 1>} {
    %0 = modelica.member_create : !modelica.member<stack, !modelica.real>
    %1 = modelica.add %arg2, %arg2 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.real>
    %3 = modelica.member_load %0 : !modelica.real
    modelica.return %3 : !modelica.real
}
