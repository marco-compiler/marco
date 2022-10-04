// RUN: modelica-opt %s --split-input-file --pass-pipeline="solve-cycles{model-name=Test process-ic-model=false debug-view=true}" | FileCheck %s

// CHECK-DAG{LITERAL}: modelica.equation_sides %16, %17 : tuple<!modelica.array<!modelica.real>>, tuple<!modelica.real>

modelica.model @Test attributes {derivatives = []} {
  %0 = modelica.member_create @x : !modelica.member<!modelica.real>
  %1 = modelica.member_create @y : !modelica.member<!modelica.real>
  %2 = modelica.member_create @z : !modelica.member<!modelica.real>
  modelica.yield %0, %1, %2 : !modelica.member<!modelica.real>, !modelica.member<!modelica.real>, !modelica.member<!modelica.real>
} body {
^bb0(%arg0: !modelica.array<!modelica.real>, %arg1: !modelica.array<!modelica.real>, %arg2: !modelica.array<!modelica.real>):
  modelica.equation attributes {match = [{indices = [[[0, 0]]], path = ["L", 0 : index, 0 : index]}]} {
    %0 = modelica.constant #modelica.int<1> : !modelica.int
    %1 = modelica.load %arg1[] : !modelica.array<!modelica.real>
    %2 = modelica.load %arg0[] : !modelica.array<!modelica.real>
    %3 = modelica.add %2, %1 : (!modelica.real, !modelica.real) -> !modelica.real
    %4 = modelica.load %arg2[] : !modelica.array<!modelica.real>
    %5 = modelica.add %3, %4 : (!modelica.real, !modelica.real) -> !modelica.real
    %6 = modelica.equation_side %5 : tuple<!modelica.real>
    %7 = modelica.equation_side %0 : tuple<!modelica.int>
    modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.equation attributes {match = [{indices = [[[0, 0]]], path = ["L", 0 : index, 1 : index]}]} {
    %0 = modelica.constant #modelica.int<2> : !modelica.int
    %1 = modelica.load %arg1[] : !modelica.array<!modelica.real>
    %2 = modelica.load %arg0[] : !modelica.array<!modelica.real>
    %3 = modelica.add %2, %1 : (!modelica.real, !modelica.real) -> !modelica.real
    %4 = modelica.load %arg2[] : !modelica.array<!modelica.real>
    %5 = modelica.sub %3, %4 : (!modelica.real, !modelica.real) -> !modelica.real
    %6 = modelica.equation_side %5 : tuple<!modelica.real>
    %7 = modelica.equation_side %0 : tuple<!modelica.int>
    modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.equation attributes {match = [{indices = [[[0, 0]]], path = ["L", 1 : index, 0 : index]}]} {
    %0 = modelica.constant #modelica.int<3> : !modelica.int
    %1 = modelica.load %arg1[] : !modelica.array<!modelica.real>
    %2 = modelica.load %arg0[] : !modelica.array<!modelica.real>
    %3 = modelica.sub %2, %1 : (!modelica.real, !modelica.real) -> !modelica.real
    %4 = modelica.load %arg2[] : !modelica.array<!modelica.real>
    %5 = modelica.add %3, %4 : (!modelica.real, !modelica.real) -> !modelica.real
    %6 = modelica.equation_side %5 : tuple<!modelica.real>
    %7 = modelica.equation_side %0 : tuple<!modelica.int>
    modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.start (%arg0 : !modelica.array<!modelica.real>) {each = false, fixed = false} {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  }
  modelica.start (%arg1 : !modelica.array<!modelica.real>) {each = false, fixed = false} {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  }
  modelica.start (%arg2 : !modelica.array<!modelica.real>) {each = false, fixed = false} {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  }
}