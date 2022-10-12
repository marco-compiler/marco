// RUN: modelica-opt %s --split-input-file --pass-pipeline="solve-cycles{model-name=Test process-ic-model=false debug-view=true}" | FileCheck %s

// CHECK{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
// CHECK: %[[#VAR:]] = modelica.load %arg0[] : !modelica.array<!modelica.real>
// CHECK-DAG: %[[#LHS:]] = modelica.equation_side %[[#VAR]] : tuple<!modelica.real>
// CHECK-DAG: %[[#LHS-1]] = modelica.constant #modelica.real<2.500000e+00> : !modelica.real
// CHECK-NEXT: %[[#LHS+1]] = modelica.equation_side %[[#LHS-1]] : tuple<!modelica.real>
// CHECK-NEXT: modelica.equation_sides %[[#LHS]], %[[#LHS+1]] : tuple<!modelica.real>, tuple<!modelica.real>

// CHECK{LITERAL}: modelica.equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
// CHECK: %[[#VAR:]] = modelica.load %arg1[] : !modelica.array<!modelica.real>
// CHECK-DAG: %[[#LHS:]] = modelica.equation_side %[[#VAR]] : tuple<!modelica.real>
// CHECK-DAG: %[[#LHS-1]] = modelica.constant #modelica.real<-1.000000e+00> : !modelica.real
// CHECK-NEXT: %[[#LHS+1]] = modelica.equation_side %[[#LHS-1]] : tuple<!modelica.real>
// CHECK-NEXT: modelica.equation_sides %[[#LHS]], %[[#LHS+1]] : tuple<!modelica.real>, tuple<!modelica.real>

// CHECK{LITERAL}: modelica.equation attributes {id = 2 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
// CHECK: %[[#VAR:]] = modelica.load %arg2[] : !modelica.array<!modelica.real>
// CHECK-DAG: %[[#LHS:]] = modelica.equation_side %[[#VAR]] : tuple<!modelica.real>
// CHECK-DAG: %[[#LHS-1]] = modelica.constant #modelica.real<-5.000000e-01> : !modelica.real
// CHECK-NEXT: %[[#LHS+1]] = modelica.equation_side %[[#LHS-1]] : tuple<!modelica.real>
// CHECK-NEXT: modelica.equation_sides %[[#LHS]], %[[#LHS+1]] : tuple<!modelica.real>, tuple<!modelica.real>

// CHECK{LITERAL}: modelica.equation attributes {id = 3 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
// CHECK: %[[#VAR:]] = modelica.load %arg3[] : !modelica.array<!modelica.real>
// CHECK: %[[#LHS:]] = modelica.equation_side %[[#VAR]] : tuple<!modelica.real>
// CHECK-NEXT: %[[#LHS+1]] = modelica.equation_side %[[#]] : tuple<!modelica.real>
// CHECK-NEXT: modelica.equation_sides %[[#LHS]], %[[#LHS+1]] : tuple<!modelica.real>, tuple<!modelica.real>

// CHECK{LITERAL}: modelica.equation attributes {id = 4 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
// CHECK: %[[#VAR:]] = modelica.load %arg4[] : !modelica.array<!modelica.real>
// CHECK: %[[#LHS:]] = modelica.equation_side %[[#VAR]] : tuple<!modelica.real>
// CHECK-NEXT: %[[#LHS+1]] = modelica.equation_side %[[#]] : tuple<!modelica.real>
// CHECK-NEXT: modelica.equation_sides %[[#LHS]], %[[#LHS+1]] : tuple<!modelica.real>, tuple<!modelica.real>

// CHECK{LITERAL}: modelica.equation attributes {id = 5 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]} {
// CHECK: %[[#VAR:]] = modelica.load %arg5[] : !modelica.array<!modelica.real>
// CHECK: %[[#LHS:]] = modelica.equation_side %[[#VAR]] : tuple<!modelica.real>
// CHECK-NEXT: %[[#LHS+1]] = modelica.equation_side %[[#]] : tuple<!modelica.real>
// CHECK-NEXT: modelica.equation_sides %[[#LHS]], %[[#LHS+1]] : tuple<!modelica.real>, tuple<!modelica.real>

modelica.model @Test attributes {derivatives = []} {
  %0 = modelica.member_create @a : !modelica.member<!modelica.real>
  %1 = modelica.member_create @b : !modelica.member<!modelica.real>
  %2 = modelica.member_create @c : !modelica.member<!modelica.real>
  %3 = modelica.member_create @x : !modelica.member<!modelica.real>
  %4 = modelica.member_create @y : !modelica.member<!modelica.real>
  %5 = modelica.member_create @z : !modelica.member<!modelica.real>
  modelica.yield %0, %1, %2, %3, %4, %5 : !modelica.member<!modelica.real>, !modelica.member<!modelica.real>, !modelica.member<!modelica.real>, !modelica.member<!modelica.real>, !modelica.member<!modelica.real>, !modelica.member<!modelica.real>
} body {
^bb0(%arg0: !modelica.array<!modelica.real>, %arg1: !modelica.array<!modelica.real>, %arg2: !modelica.array<!modelica.real>, %arg3: !modelica.array<!modelica.real>, %arg4: !modelica.array<!modelica.real>, %arg5: !modelica.array<!modelica.real>):
  modelica.equation attributes {id = 0, match = [{indices = [[[0, 0]]], path = ["L", 0 : index, 0 : index]}]} {
    %0 = modelica.load %arg1[] : !modelica.array<!modelica.real>
    %1 = modelica.load %arg0[] : !modelica.array<!modelica.real>
    %2 = modelica.add %1, %0 : (!modelica.real, !modelica.real) -> !modelica.real
    %3 = modelica.load %arg2[] : !modelica.array<!modelica.real>
    %4 = modelica.add %2, %3 : (!modelica.real, !modelica.real) -> !modelica.real
    %5 = modelica.constant #modelica.int<1> : !modelica.int
    %6 = modelica.equation_side %4 : tuple<!modelica.real>
    %7 = modelica.equation_side %5 : tuple<!modelica.int>
    modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.equation attributes {id = 1, match = [{indices = [[[0, 0]]], path = ["L", 0 : index, 1 : index]}]} {
    %0 = modelica.load %arg1[] : !modelica.array<!modelica.real>
    %1 = modelica.load %arg0[] : !modelica.array<!modelica.real>
    %2 = modelica.add %1, %0 : (!modelica.real, !modelica.real) -> !modelica.real
    %3 = modelica.load %arg2[] : !modelica.array<!modelica.real>
    %4 = modelica.sub %2, %3 : (!modelica.real, !modelica.real) -> !modelica.real
    %5 = modelica.constant #modelica.int<2> : !modelica.int
    %6 = modelica.equation_side %4 : tuple<!modelica.real>
    %7 = modelica.equation_side %5 : tuple<!modelica.int>
    modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.equation attributes {id = 2, match = [{indices = [[[0, 0]]], path = ["L", 1 : index]}]} {
    %0 = modelica.load %arg1[] : !modelica.array<!modelica.real>
    %1 = modelica.load %arg0[] : !modelica.array<!modelica.real>
    %2 = modelica.sub %1, %0 : (!modelica.real, !modelica.real) -> !modelica.real
    %3 = modelica.load %arg2[] : !modelica.array<!modelica.real>
    %4 = modelica.add %2, %3 : (!modelica.real, !modelica.real) -> !modelica.real
    %5 = modelica.constant #modelica.int<3> : !modelica.int
    %6 = modelica.equation_side %4 : tuple<!modelica.real>
    %7 = modelica.equation_side %5 : tuple<!modelica.int>
    modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.equation attributes {id = 3, match = [{indices = [[[0, 0]]], path = ["L", 0 : index, 0 : index, 0 : index, 0 : index]}]} {
    %0 = modelica.load %arg4[] : !modelica.array<!modelica.real>
    %1 = modelica.load %arg3[] : !modelica.array<!modelica.real>
    %2 = modelica.add %1, %0 : (!modelica.real, !modelica.real) -> !modelica.real
    %3 = modelica.load %arg5[] : !modelica.array<!modelica.real>
    %4 = modelica.add %2, %3 : (!modelica.real, !modelica.real) -> !modelica.real
    %5 = modelica.load %arg1[] : !modelica.array<!modelica.real>
    %6 = modelica.add %4, %5 : (!modelica.real, !modelica.real) -> !modelica.real
    %7 = modelica.constant #modelica.int<2> : !modelica.int
    %8 = modelica.load %arg2[] : !modelica.array<!modelica.real>
    %9 = modelica.mul %7, %8 : (!modelica.int, !modelica.real) -> !modelica.real
    %10 = modelica.sub %6, %9 : (!modelica.real, !modelica.real) -> !modelica.real
    %11 = modelica.constant #modelica.int<1> : !modelica.int
    %12 = modelica.equation_side %10 : tuple<!modelica.real>
    %13 = modelica.equation_side %11 : tuple<!modelica.int>
    modelica.equation_sides %12, %13 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.equation attributes {id = 4, match = [{indices = [[[0, 0]]], path = ["L", 0 : index, 1 : index]}]} {
    %0 = modelica.load %arg4[] : !modelica.array<!modelica.real>
    %1 = modelica.load %arg3[] : !modelica.array<!modelica.real>
    %2 = modelica.add %1, %0 : (!modelica.real, !modelica.real) -> !modelica.real
    %3 = modelica.load %arg5[] : !modelica.array<!modelica.real>
    %4 = modelica.sub %2, %3 : (!modelica.real, !modelica.real) -> !modelica.real
    %5 = modelica.constant #modelica.int<2> : !modelica.int
    %6 = modelica.equation_side %4 : tuple<!modelica.real>
    %7 = modelica.equation_side %5 : tuple<!modelica.int>
    modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.equation attributes {id = 5, match = [{indices = [[[0, 0]]], path = ["L", 1 : index]}]} {
    %0 = modelica.load %arg4[] : !modelica.array<!modelica.real>
    %1 = modelica.load %arg3[] : !modelica.array<!modelica.real>
    %2 = modelica.sub %1, %0 : (!modelica.real, !modelica.real) -> !modelica.real
    %3 = modelica.load %arg5[] : !modelica.array<!modelica.real>
    %4 = modelica.add %2, %3 : (!modelica.real, !modelica.real) -> !modelica.real
    %5 = modelica.constant #modelica.int<3> : !modelica.int
    %6 = modelica.equation_side %4 : tuple<!modelica.real>
    %7 = modelica.equation_side %5 : tuple<!modelica.int>
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
  modelica.start (%arg3 : !modelica.array<!modelica.real>) {each = false, fixed = false} {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  }
  modelica.start (%arg4 : !modelica.array<!modelica.real>) {each = false, fixed = false} {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  }
  modelica.start (%arg5 : !modelica.array<!modelica.real>) {each = false, fixed = false} {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  }
}
