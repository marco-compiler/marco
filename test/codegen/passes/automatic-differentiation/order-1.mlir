// RUN: modelica-opt %s --auto-diff | FileCheck %s

modelica.function @foo(%arg0 : !modelica.ptr<2x!modelica.real>, %arg1 : !modelica.int, %arg2 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y", "z"], results_names = ["t"], derivative = #modelica.derivative<"foo1", 1>} {
    %0 = modelica.member_create {name = "t"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.add %arg2, %arg2 : (!modelica.real, !modelica.real) -> !modelica.real
    %2 = modelica.constant #modelica.int<2>
    %3 = modelica.mul %1, %2 : (!modelica.real, !modelica.int) -> !modelica.real
    modelica.member_store %0, %3 : !modelica.member<stack, !modelica.real>
    %4 = modelica.member_load %0 : !modelica.real
    modelica.return %4 : !modelica.real
}
