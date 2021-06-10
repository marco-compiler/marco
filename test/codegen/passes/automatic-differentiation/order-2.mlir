module  {
  modelica.function @foo1(%arg0 : !modelica.ptr<2x!modelica.real>, %arg1 : !modelica.int, %arg2 : !modelica.real, %arg3 : !modelica.ptr<2x!modelica.real>, %arg4 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y", "z", "der_x", "der_z"], derivative = #modelica.derivative<"foo2", 2>, results_names = ["der_t"]} {
    %0 = modelica.member_create {name = "t"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.member_create {name = "der_t"} : !modelica.member<stack, !modelica.real>
    %2 = modelica.add %arg4, %arg4 : (!modelica.real, !modelica.real) -> !modelica.real
    %3 = modelica.add %arg2, %arg2 : (!modelica.real, !modelica.real) -> !modelica.real
    %4 = modelica.constant #modelica.real<0.000000>
    %5 = modelica.constant #modelica.int<2>
    %6 = modelica.mul %2, %5 : (!modelica.real, !modelica.int) -> !modelica.real
    %7 = modelica.mul %3, %4 : (!modelica.real, !modelica.real) -> !modelica.real
    %8 = modelica.add %6, %7 : (!modelica.real, !modelica.real) -> !modelica.real
    %9 = modelica.mul %3, %5 : (!modelica.real, !modelica.int) -> !modelica.real
    modelica.member_store %1, %8 : !modelica.member<stack, !modelica.real>
    modelica.member_store %0, %9 : !modelica.member<stack, !modelica.real>
    %10 = modelica.member_load %1 : !modelica.real
    %11 = modelica.member_load %0 : !modelica.real
    modelica.return %10 : !modelica.real
  }
  modelica.function @foo(%arg0 : !modelica.ptr<2x!modelica.real>, %arg1 : !modelica.int, %arg2 : !modelica.real) -> (!modelica.real) attributes {args_names = ["x", "y", "z"], derivative = #modelica.derivative<"foo1", 1>, results_names = ["t"]} {
    %0 = modelica.member_create {name = "t"} : !modelica.member<stack, !modelica.real>
    %1 = modelica.add %arg2, %arg2 : (!modelica.real, !modelica.real) -> !modelica.real
    modelica.member_store %0, %1 : !modelica.member<stack, !modelica.real>
    %2 = modelica.member_load %0 : !modelica.real
    modelica.return %2 : !modelica.real
  }
}