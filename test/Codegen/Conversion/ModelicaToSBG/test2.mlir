// -----------------------------------------------------------------------------
// model Test2
//   //Model with possibly wrong initial matching
//   constant Integer N = 100;
//   Real a[N], x[N], b[N];
// equation
//   for i in 2:N-1 loop
//     der(x[i]) = a[i] - x[i];
//     a[i + 1] = a[i] + b[i];
//     b[i] = x[i - 1];
//   end for;
//   b[1] = 0;
//   b[N] = x[N - 1];
//   der(x[1]) = a[1] - x[1];
//   der(x[N]) = a[N] - x[N];
//   a[2] = a[1] + b[1];
//   a[N] = 1;
// end Test2;
// -----------------------------------------------------------------------------

module {
 modelica.model @test attributes {derivatives_map = [#modelica<var_derivative @x, @der_x, {[0,99]}>]} {
    modelica.variable @a : !modelica.variable<100x!modelica.real>
    modelica.variable @b : !modelica.variable<100x!modelica.real>
    modelica.variable @x : !modelica.variable<100x!modelica.real>
    modelica.variable @der_x : !modelica.variable<100x!modelica.real>

    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
      %0 = modelica.variable_get @a : !modelica.array<100x!modelica.real>
      %1 = modelica.variable_get @x : !modelica.array<100x!modelica.real>
      %2 = modelica.variable_get @der_x : !modelica.array<100x!modelica.real>
      %3 = modelica.load %0[%i0] : !modelica.array<100x!modelica.real>
      %4 = modelica.load %1[%i0] : !modelica.array<100x!modelica.real>
      %5 = modelica.load %2[%i0] : !modelica.array<100x!modelica.real>
      %6 = modelica.sub %3, %4 : (!modelica.real, !modelica.real) -> !modelica.real
      %7 = modelica.equation_side %5 : tuple<!modelica.real>
      %8 = modelica.equation_side %6 : tuple<!modelica.real>
      modelica.equation_sides %7, %8 : tuple<!modelica.real>, tuple<!modelica.real>
    }
    modelica.equation_instance %t0 {indices = #modeling<multidim_range [1,98]>} : !modelica.equation

    %t1 = modelica.equation_template inductions = [%i0] attributes {id = "t1"} {
      %0 = modelica.variable_get @a : !modelica.array<100x!modelica.real>
      %1 = modelica.variable_get @b : !modelica.array<100x!modelica.real>
      %2 = modelica.load %0[%i0] : !modelica.array<100x!modelica.real>
      %3 = modelica.load %0[%i0] : !modelica.array<100x!modelica.real>
      %4 = modelica.load %1[%i0] : !modelica.array<100x!modelica.real>
      %5 = modelica.add %2, %4 : (!modelica.real, !modelica.real) -> !modelica.real
      %6 = modelica.equation_side %3 : tuple<!modelica.real>
      %7 = modelica.equation_side %5 : tuple<!modelica.real>
      modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.real>
    }
    modelica.equation_instance %t1 {indices = #modeling<multidim_range [1,98]>} : !modelica.equation

    %t2 = modelica.equation_template inductions = [%i0] attributes {id = "t2"} {
      %0 = modelica.variable_get @b : !modelica.array<100x!modelica.real>
      %1 = modelica.variable_get @x : !modelica.array<100x!modelica.real>
      %2 = modelica.load %0[%i0] : !modelica.array<100x!modelica.real>
      %3 = index.constant 1
      %4 = index.sub %i0, %3
      %5 = modelica.load %1[%4] : !modelica.array<100x!modelica.real>
      %6 = modelica.equation_side %2 : tuple<!modelica.real>
      %7 = modelica.equation_side %5 : tuple<!modelica.real>
      modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.real>
    }
    modelica.equation_instance %t2 {indices = #modeling<multidim_range [1,98]>} : !modelica.equation
 }
}