// RUN: modelica-opt %s --split-input-file --pass-pipeline="solve-cycles{model-name=Test process-ic-model=false debug-view=true}" | FileCheck %s

// CHECK-DAG{LITERAL}: modelica.equation_sides %16, %17 : tuple<!modelica.array<!modelica.real>>, tuple<!modelica.real>

modelica.model @Test attributes {derivatives = []} {
  %0 = modelica.member_create @x : !modelica.member<7x!modelica.real>
  modelica.yield %0 : !modelica.member<7x!modelica.real>
} body {
^bb0(%arg0: !modelica.array<7x!modelica.real>):
  modelica.equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L", 0 : index, 0 : index, 1 : index]}]} {
    %0 = modelica.constant #modelica.int<1> : !modelica.int
    %1 = modelica.constant 1 : index
    %2 = modelica.constant -1 : index
    %3 = modelica.constant 0 : index
    %4 = modelica.constant #modelica.int<2> : !modelica.int
    %5 = modelica.constant 6 : index
    %6 = modelica.constant 5 : index
    %7 = modelica.constant #modelica.int<7> : !modelica.int
    %8 = modelica.subscription %arg0[%3] : !modelica.array<7x!modelica.real>
    %9 = modelica.subscription %arg0[%1] : !modelica.array<7x!modelica.real>
    %10 = modelica.load %9[] : !modelica.array<!modelica.real>
    %11 = modelica.load %8[] : !modelica.array<!modelica.real>
    %12 = modelica.add %11, %10 : (!modelica.real, !modelica.real) -> !modelica.real
    %13 = modelica.subscription %arg0[%6] : !modelica.array<7x!modelica.real>
    %14 = modelica.load %13[] : !modelica.array<!modelica.real>
    %15 = modelica.add %12, %14 : (!modelica.real, !modelica.real) -> !modelica.real
    %16 = modelica.subscription %arg0[%5] : !modelica.array<7x!modelica.real>
    %17 = modelica.load %16[] : !modelica.array<!modelica.real>
    %18 = modelica.add %15, %17 : (!modelica.real, !modelica.real) -> !modelica.real
    %19 = modelica.equation_side %18 : tuple<!modelica.real>
    %20 = modelica.equation_side %0 : tuple<!modelica.int>
    modelica.equation_sides %19, %20 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.equation attributes {id = 1, match = [{indices = [[[0, 0]]], path = ["L", 0 : index, 0 : index, 0 : index]}]} {
    %0 = modelica.constant #modelica.int<2> : !modelica.int
    %1 = modelica.constant 1 : index
    %2 = modelica.constant -1 : index
    %3 = modelica.constant 0 : index
    %4 = modelica.constant #modelica.int<2> : !modelica.int
    %5 = modelica.constant 6 : index
    %6 = modelica.constant 5 : index
    %7 = modelica.constant #modelica.int<7> : !modelica.int
    %8 = modelica.subscription %arg0[%3] : !modelica.array<7x!modelica.real>
    %9 = modelica.subscription %arg0[%1] : !modelica.array<7x!modelica.real>
    %10 = modelica.load %9[] : !modelica.array<!modelica.real>
    %11 = modelica.load %8[] : !modelica.array<!modelica.real>
    %12 = modelica.add %11, %10 : (!modelica.real, !modelica.real) -> !modelica.real
    %13 = modelica.subscription %arg0[%6] : !modelica.array<7x!modelica.real>
    %14 = modelica.load %13[] : !modelica.array<!modelica.real>
    %15 = modelica.add %12, %14 : (!modelica.real, !modelica.real) -> !modelica.real
    %16 = modelica.subscription %arg0[%5] : !modelica.array<7x!modelica.real>
    %17 = modelica.load %16[] : !modelica.array<!modelica.real>
    %18 = modelica.sub %15, %17 : (!modelica.real, !modelica.real) -> !modelica.real
    %19 = modelica.equation_side %18 : tuple<!modelica.real>
    %20 = modelica.equation_side %0 : tuple<!modelica.int>
    modelica.equation_sides %19, %20 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.equation attributes {id = 2, match = [{indices = [[[0, 0]]], path = ["L", 0 : index, 1 : index]}]} {
    %0 = modelica.constant #modelica.int<3> : !modelica.int
    %1 = modelica.constant 1 : index
    %2 = modelica.constant -1 : index
    %3 = modelica.constant 0 : index
    %4 = modelica.constant #modelica.int<2> : !modelica.int
    %5 = modelica.constant 6 : index
    %6 = modelica.constant 5 : index
    %7 = modelica.constant #modelica.int<7> : !modelica.int
    %8 = modelica.subscription %arg0[%3] : !modelica.array<7x!modelica.real>
    %9 = modelica.subscription %arg0[%1] : !modelica.array<7x!modelica.real>
    %10 = modelica.load %9[] : !modelica.array<!modelica.real>
    %11 = modelica.load %8[] : !modelica.array<!modelica.real>
    %12 = modelica.add %11, %10 : (!modelica.real, !modelica.real) -> !modelica.real
    %13 = modelica.subscription %arg0[%6] : !modelica.array<7x!modelica.real>
    %14 = modelica.load %13[] : !modelica.array<!modelica.real>
    %15 = modelica.sub %12, %14 : (!modelica.real, !modelica.real) -> !modelica.real
    %16 = modelica.subscription %arg0[%5] : !modelica.array<7x!modelica.real>
    %17 = modelica.load %16[] : !modelica.array<!modelica.real>
    %18 = modelica.add %15, %17 : (!modelica.real, !modelica.real) -> !modelica.real
    %19 = modelica.equation_side %18 : tuple<!modelica.real>
    %20 = modelica.equation_side %0 : tuple<!modelica.int>
    modelica.equation_sides %19, %20 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.equation attributes {id = 3, match = [{indices = [[[0, 0]]], path = ["L", 1 : index]}]} {
    %0 = modelica.constant #modelica.int<4> : !modelica.int
    %1 = modelica.constant 1 : index
    %2 = modelica.constant -1 : index
    %3 = modelica.constant 0 : index
    %4 = modelica.constant #modelica.int<2> : !modelica.int
    %5 = modelica.constant 6 : index
    %6 = modelica.constant 5 : index
    %7 = modelica.constant #modelica.int<7> : !modelica.int
    %8 = modelica.subscription %arg0[%3] : !modelica.array<7x!modelica.real>
    %9 = modelica.subscription %arg0[%1] : !modelica.array<7x!modelica.real>
    %10 = modelica.load %9[] : !modelica.array<!modelica.real>
    %11 = modelica.load %8[] : !modelica.array<!modelica.real>
    %12 = modelica.sub %11, %10 : (!modelica.real, !modelica.real) -> !modelica.real
    %13 = modelica.subscription %arg0[%6] : !modelica.array<7x!modelica.real>
    %14 = modelica.load %13[] : !modelica.array<!modelica.real>
    %15 = modelica.add %12, %14 : (!modelica.real, !modelica.real) -> !modelica.real
    %16 = modelica.subscription %arg0[%5] : !modelica.array<7x!modelica.real>
    %17 = modelica.load %16[] : !modelica.array<!modelica.real>
    %18 = modelica.add %15, %17 : (!modelica.real, !modelica.real) -> !modelica.real
    %19 = modelica.equation_side %18 : tuple<!modelica.real>
    %20 = modelica.equation_side %0 : tuple<!modelica.int>
    modelica.equation_sides %19, %20 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.for_equation %arg1 = 1 to 3 {
    modelica.equation attributes {id = 4, match = [{indices = [[[2, 3]]], path = ["L", 0 : index, 0 : index, 0 : index, 1 : index]}, {indices = [[[1, 1]]], path = ["L", 1 : index]}]} {
      %0 = modelica.constant #modelica.int<1> : !modelica.int
      %1 = modelica.constant -1 : index
      %2 = modelica.constant 1 : index
      %3 = modelica.constant 2 : index
      %4 = modelica.constant 3 : index
      %5 = modelica.constant 4 : index
      %6 = modelica.add %arg1, %1 : (index, index) -> index
      %7 = modelica.subscription %arg0[%6] : !modelica.array<7x!modelica.real>
      %8 = modelica.add %arg1, %2 : (index, index) -> index
      %9 = modelica.add %8, %1 : (index, index) -> index
      %10 = modelica.subscription %arg0[%9] : !modelica.array<7x!modelica.real>
      %11 = modelica.load %10[] : !modelica.array<!modelica.real>
      %12 = modelica.load %7[] : !modelica.array<!modelica.real>
      %13 = modelica.add %12, %11 : (!modelica.real, !modelica.real) -> !modelica.real
      %14 = modelica.add %arg1, %3 : (index, index) -> index
      %15 = modelica.add %14, %1 : (index, index) -> index
      %16 = modelica.subscription %arg0[%15] : !modelica.array<7x!modelica.real>
      %17 = modelica.load %16[] : !modelica.array<!modelica.real>
      %18 = modelica.add %13, %17 : (!modelica.real, !modelica.real) -> !modelica.real
      %19 = modelica.add %arg1, %4 : (index, index) -> index
      %20 = modelica.add %19, %1 : (index, index) -> index
      %21 = modelica.subscription %arg0[%20] : !modelica.array<7x!modelica.real>
      %22 = modelica.load %21[] : !modelica.array<!modelica.real>
      %23 = modelica.sub %18, %22 : (!modelica.real, !modelica.real) -> !modelica.real
      %24 = modelica.add %arg1, %5 : (index, index) -> index
      %25 = modelica.add %24, %1 : (index, index) -> index
      %26 = modelica.subscription %arg0[%25] : !modelica.array<7x!modelica.real>
      %27 = modelica.load %26[] : !modelica.array<!modelica.real>
      %28 = modelica.add %23, %27 : (!modelica.real, !modelica.real) -> !modelica.real
      %29 = modelica.equation_side %28 : tuple<!modelica.real>
      %30 = modelica.equation_side %0 : tuple<!modelica.int>
      modelica.equation_sides %29, %30 : tuple<!modelica.real>, tuple<!modelica.int>
    }
  }
  modelica.start (%arg0 : !modelica.array<7x!modelica.real>) {each = true, fixed = false} {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  }
}
