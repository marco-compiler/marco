// -----------------------------------------------------------------------------
// model test
//   Real a[100];
// equation
//   sum(a[:]) = 0;
// end test;
// -----------------------------------------------------------------------------

module {
  modelica.model @test {
    modelica.variable @a : !modelica.variable<100x!modelica.real>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
      %0 = modelica.variable_get @a : !modelica.array<100x!modelica.real>
      %1 = modelica.sum %0 : !modelica.array<100x!modelica.real> -> !modelica.real
      %2 = modelica.constant #modelica.real<0.0>
      %3 = modelica.equation_side %2 : tuple<!modelica.real>
      %4 = modelica.equation_side %1 : tuple<!modelica.real>
      modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }
    modelica.equation_instance %t0 : !modelica.equation
  }
}