// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(match{model-name=Test process-ic-model=false debug-view=true})" | FileCheck %s

// x = 0;
// x = y;
// y = z;

// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 0 : i64, match = [{indices = [[[0, 0]]], path = ["L"]}]}
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 1 : i64, match = [{indices = [[[0, 0]]], path = ["R"]}]}
// CHECK-DAG{LITERAL}: modelica.equation attributes {id = 2 : i64, match = [{indices = [[[0, 0]]], path = ["R"]}]}

modelica.model @TestImplicit  attributes {derivatives = []}{
  modelica.variable @v1 : !modelica.variable<!modelica.real>
  modelica.variable @v2 : !modelica.variable<!modelica.real>
  modelica.variable @v3 : !modelica.variable<!modelica.real>
  modelica.variable @i1 : !modelica.variable<!modelica.real>
  modelica.variable @i2 : !modelica.variable<!modelica.real>
  modelica.variable @i3 : !modelica.variable<!modelica.real>
  modelica.variable @v : !modelica.variable<!modelica.real>
  modelica.variable @R1 : !modelica.variable<!modelica.real, parameter>
  modelica.variable @R2 : !modelica.variable<!modelica.real, parameter>
  modelica.variable @R3 : !modelica.variable<!modelica.real, parameter>

  modelica.start @R1 {
    %0 = modelica.constant #modelica.real<1.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  } {each = false, fixed = true}

  modelica.start @R2 {
    %0 = modelica.constant #modelica.real<2.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  } {each = false, fixed = true}

  modelica.start @R3 {
    %0 = modelica.constant #modelica.real<2.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  } {each = false, fixed = true}

  modelica.equation attributes {match = [{indices = [[[0, 0]]], path = ["L"]}]} {
    %0 = modelica.variable_get @v : !modelica.real
    %1 = modelica.time : !modelica.real
    %2 = modelica.sin %1 : !modelica.real -> !modelica.real
    %3 = modelica.equation_side %0 : tuple<!modelica.real>
    %4 = modelica.equation_side %2 : tuple<!modelica.real>
    modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
  }

  modelica.equation attributes {match = [{indices = [[[0, 0]]], path = ["L"]}]} {
    %0 = modelica.variable_get @i2 : !modelica.real
    %1 = modelica.variable_get @i1 : !modelica.real
    %2 = modelica.neg %1 : !modelica.real -> !modelica.real
    %3 = modelica.equation_side %0 : tuple<!modelica.real>
    %4 = modelica.equation_side %2 : tuple<!modelica.real>
    modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
  }

  modelica.equation attributes {match = [{indices = [[[0, 0]]], path = ["L"]}]} {
    %0 = modelica.variable_get @i3 : !modelica.real
    %1 = modelica.variable_get @i2 : !modelica.real
    %2 = modelica.neg %1 : !modelica.real -> !modelica.real
    %3 = modelica.equation_side %0 : tuple<!modelica.real>
    %4 = modelica.equation_side %2 : tuple<!modelica.real>
    modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
  }

  modelica.equation attributes {match = [{indices = [[[0, 0]]], path = ["R", 1 : index]}]} {
    %0 = modelica.variable_get @v1 : !modelica.real
    %1 = modelica.variable_get @R1 : !modelica.real
    %2 = modelica.variable_get @i1 : !modelica.real
    %3 = modelica.mul %1, %2 : (!modelica.real, !modelica.real) -> !modelica.real
    %4 = modelica.equation_side %0 : tuple<!modelica.real>
    %5 = modelica.equation_side %3 : tuple<!modelica.real>
    modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
  }

  modelica.equation attributes {match = [{indices = [[[0, 0]]], path = ["L"]}]} {
    %0 = modelica.variable_get @v2 : !modelica.real
    %1 = modelica.variable_get @R2 : !modelica.real
    %2 = modelica.variable_get @i2 : !modelica.real
    %3 = modelica.mul %1, %2 : (!modelica.real, !modelica.real) -> !modelica.real
    %4 = modelica.neg %3 : !modelica.real -> !modelica.real
    %5 = modelica.equation_side %0 : tuple<!modelica.real>
    %6 = modelica.equation_side %4 : tuple<!modelica.real>
    modelica.equation_sides %5, %6 : tuple<!modelica.real>, tuple<!modelica.real>
  }

  modelica.equation attributes {match = [{indices = [[[0, 0]]], path = ["L"]}]} {
    %0 = modelica.variable_get @v3 : !modelica.real
    %1 = modelica.variable_get @R3 : !modelica.real
    %2 = modelica.variable_get @i3 : !modelica.real
    %3 = modelica.mul %1, %2 : (!modelica.real, !modelica.real) -> !modelica.real
    %4 = modelica.equation_side %0 : tuple<!modelica.real>
    %5 = modelica.equation_side %3 : tuple<!modelica.real>
    modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
  }

  modelica.equation attributes {match = [{indices = [[[0, 0]]], path = ["R", 0 : index, 0 : index]}]} {
    %0 = modelica.variable_get @v : !modelica.real
    %1 = modelica.variable_get @v1 : !modelica.real
    %2 = modelica.variable_get @v2 : !modelica.real
    %3 = modelica.add %1, %2 : (!modelica.real, !modelica.real) -> !modelica.real
    %4 = modelica.variable_get @v3 : !modelica.real
    %5 = modelica.add %3, %4 : (!modelica.real, !modelica.real) -> !modelica.real
    %6 = modelica.equation_side %0 : tuple<!modelica.real>
    %7 = modelica.equation_side %5 : tuple<!modelica.real>
    modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.real>
  }

  modelica.start @v1 {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  } {each = false, fixed = false}

  modelica.start @v2 {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  } {each = false, fixed = false}

  modelica.start @v3 {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  } {each = false, fixed = false}

  modelica.start @i1 {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  } {each = false, fixed = false}

  modelica.start @i2 {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  } {each = false, fixed = false}

  modelica.start @i3 {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  } {each = false, fixed = false}

  modelica.start @v {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  } {each = false, fixed = false}

  modelica.initial_equation attributes {match = [{indices = [[[0, 0]]], path = ["L"]}]} {
    %0 = modelica.variable_get @R1 : !modelica.real
    %1 = modelica.constant #modelica.real<1.000000e+00> : !modelica.real
    %2 = modelica.equation_side %0 : tuple<!modelica.real>
    %3 = modelica.equation_side %1 : tuple<!modelica.real>
    modelica.equation_sides %2, %3 : tuple<!modelica.real>, tuple<!modelica.real>
  }

  modelica.initial_equation attributes {match = [{indices = [[[0, 0]]], path = ["L"]}]} {
    %0 = modelica.variable_get @R2 : !modelica.real
    %1 = modelica.constant #modelica.real<2.000000e+00> : !modelica.real
    %2 = modelica.equation_side %0 : tuple<!modelica.real>
    %3 = modelica.equation_side %1 : tuple<!modelica.real>
    modelica.equation_sides %2, %3 : tuple<!modelica.real>, tuple<!modelica.real>
  }

  modelica.initial_equation attributes {match = [{indices = [[[0, 0]]], path = ["L"]}]} {
    %0 = modelica.variable_get @R3 : !modelica.real
    %1 = modelica.constant #modelica.real<2.000000e+00> : !modelica.real
    %2 = modelica.equation_side %0 : tuple<!modelica.real>
    %3 = modelica.equation_side %1 : tuple<!modelica.real>
    modelica.equation_sides %2, %3 : tuple<!modelica.real>, tuple<!modelica.real>
  }
}