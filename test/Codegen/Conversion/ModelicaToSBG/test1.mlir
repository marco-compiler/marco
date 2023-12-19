// -----------------------------------------------------------------------------
//  model test
//    constant Integer N = 1000000;
//    Real a[N], b[N], x[N];
//  equation
//    for i in 1:N loop
//      a[i] = 2 * x[i] - b[i];
//      a[i] = 2 * b[i] - x[i];
//      der(x[i]) = 1 - a[i];
//    end for;
//  end test;
// -----------------------------------------------------------------------------

module {
 modelica.model @test attributes {derivatives_map = [#modelica<var_derivative @x, @der_x, {[0,999999]}>]} {
    modelica.variable @a : !modelica.variable<1000000x!modelica.real>
    modelica.variable @b : !modelica.variable<1000000x!modelica.real>
    modelica.variable @x : !modelica.variable<1000000x!modelica.real>
    modelica.variable @der_x : !modelica.variable<1000000x!modelica.real>

    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
      %0 = modelica.variable_get @a : !modelica.array<1000000x!modelica.real>
      %1 = modelica.variable_get @b : !modelica.array<1000000x!modelica.real>
      %2 = modelica.variable_get @x : !modelica.array<1000000x!modelica.real>
      %3 = modelica.load %0[%i0] : !modelica.array<1000000x!modelica.real>
      %4 = modelica.load %1[%i0] : !modelica.array<1000000x!modelica.real>
      %5 = modelica.load %2[%i0] : !modelica.array<1000000x!modelica.real>
      %6 = modelica.constant #modelica.real<2.0>
      %7 = modelica.mul %6, %5 : (!modelica.real, !modelica.real) -> !modelica.real
      %8 = modelica.sub %5, %4 : (!modelica.real, !modelica.real) -> !modelica.real
      %9 = modelica.equation_side %3 : tuple<!modelica.real>
      %10 = modelica.equation_side %8 : tuple<!modelica.real>
      modelica.equation_sides %9, %10 : tuple<!modelica.real>, tuple<!modelica.real>
    }
    modelica.equation_instance %t0 {indices = #modeling<multidim_range [0,999999]>} : !modelica.equation

    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
      %0 = modelica.variable_get @a : !modelica.array<1000000x!modelica.real>
      %1 = modelica.variable_get @b : !modelica.array<1000000x!modelica.real>
      %2 = modelica.variable_get @x : !modelica.array<1000000x!modelica.real>
      %3 = modelica.load %0[%i0] : !modelica.array<1000000x!modelica.real>
      %4 = modelica.load %1[%i0] : !modelica.array<1000000x!modelica.real>
      %5 = modelica.load %2[%i0] : !modelica.array<1000000x!modelica.real>
      %6 = modelica.constant #modelica.real<2.0>
      %7 = modelica.mul %6, %4 : (!modelica.real, !modelica.real) -> !modelica.real
      %8 = modelica.sub %7, %5 : (!modelica.real, !modelica.real) -> !modelica.real
      %9 = modelica.equation_side %3 : tuple<!modelica.real>
      %10 = modelica.equation_side %8 : tuple<!modelica.real>
      modelica.equation_sides %9, %10 : tuple<!modelica.real>, tuple<!modelica.real>
    }
    modelica.equation_instance %t1 {indices = #modeling<multidim_range [0,999999]>} : !modelica.equation

    %t2 = modelica.equation_template inductions = [%i0] attributes {id = "t2"} {
      %0 = modelica.variable_get @der_x : !modelica.array<1000000x!modelica.real>
      %1 = modelica.variable_get @a : !modelica.array<1000000x!modelica.real>
      %2 = modelica.load %0[%i0] : !modelica.array<1000000x!modelica.real>
      %3 = modelica.load %1[%i0] : !modelica.array<1000000x!modelica.real>
      %5 = modelica.constant #modelica.real<1.0>
      %6 = modelica.sub %5, %3 : (!modelica.real, !modelica.real) -> !modelica.real
      %7 = modelica.equation_side %2 : tuple<!modelica.real>
      %8 = modelica.equation_side %6 : tuple<!modelica.real>
      modelica.equation_sides %7, %8 : tuple<!modelica.real>, tuple<!modelica.real>
    }
    modelica.equation_instance %t2 {indices = #modeling<multidim_range [0,999999]>} : !modelica.equation
 }
}