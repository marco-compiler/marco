// RUN: modelica-opt %s --split-input-file --pass-pipeline="solve-cycles{model-name=Test process-ic-model=false debug-view=true}" | FileCheck %s

// CHECK-DAG{LITERAL}: modelica.equation_sides %16, %17 : tuple<!modelica.array<!modelica.real>>, tuple<!modelica.real>

modelica.model @Test attributes {derivatives = []} {
  %0 = modelica.member_create @x : !modelica.member<3x!modelica.real>
  modelica.yield %0 : !modelica.member<3x!modelica.real>
} body {
^bb0(%arg0: !modelica.array<3x!modelica.real>):
  modelica.equation attributes {match = [{indices = [[[0, 0]]], path = ["L", 1 : index]}]} {
    %0 = modelica.constant #modelica.int<0> : !modelica.int
    %1 = modelica.constant 1 : index
    %2 = modelica.constant -1 : index
    %3 = modelica.constant 0 : index
    %4 = modelica.constant 2 : index
    %5 = modelica.constant #modelica.int<3> : !modelica.int
    %6 = modelica.subscription %arg0[%3] : !modelica.array<3x!modelica.real>
    %7 = modelica.subscription %arg0[%1] : !modelica.array<3x!modelica.real>
    %8 = modelica.load %7[] : !modelica.array<!modelica.real>
    %9 = modelica.load %6[] : !modelica.array<!modelica.real>
    %10 = modelica.add %9, %8 : (!modelica.real, !modelica.real) -> !modelica.real
    %11 = modelica.subscription %arg0[%4] : !modelica.array<3x!modelica.real>
    %12 = modelica.load %11[] : !modelica.array<!modelica.real>
    %13 = modelica.add %10, %12 : (!modelica.real, !modelica.real) -> !modelica.real
    %14 = modelica.equation_side %13 : tuple<!modelica.real>
    %15 = modelica.equation_side %0 : tuple<!modelica.int>
    modelica.equation_sides %14, %15 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.for_equation %arg1 = 1 to 2 {
    modelica.initial_equation attributes {match = [{indices = [[[2, 2]]], path = ["L", 0 : index]}, {indices = [[[1, 1]]], path = ["L", 0 : index]}]} {
      %0 = modelica.constant #modelica.int<1> : !modelica.int
      %1 = modelica.constant -1 : index
      %2 = modelica.constant 1 : index
      %3 = modelica.add %arg1, %1 : (index, index) -> index
      %4 = modelica.subscription %arg0[%3] : !modelica.array<3x!modelica.real>
      %5 = modelica.add %arg1, %2 : (index, index) -> index
      %6 = modelica.add %5, %1 : (index, index) -> index
      %7 = modelica.subscription %arg0[%6] : !modelica.array<3x!modelica.real>
      %8 = modelica.load %7[] : !modelica.array<!modelica.real>
      %9 = modelica.load %4[] : !modelica.array<!modelica.real>
      %10 = modelica.add %9, %8 : (!modelica.real, !modelica.real) -> !modelica.real
      %11 = modelica.equation_side %10 : tuple<!modelica.real>
      %12 = modelica.equation_side %0 : tuple<!modelica.int>
      modelica.equation_sides %11, %12 : tuple<!modelica.real>, tuple<!modelica.int>
    }
  }
  modelica.start (%arg0 : !modelica.array<3x!modelica.real>) {each = true, fixed = false} {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  }
}
