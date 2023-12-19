// -----------------------------------------------------------------------------
// model test
//   Real a[100, 100], b[100];
// equation
//   for i in [0:99] loop
//    a[i] = b;
//    b[i] = 0;
//   end loop
// end test;
// -----------------------------------------------------------------------------

module {
 modelica.model @test {
    modelica.variable @a : !modelica.variable<100x100x!modelica.real>
    modelica.variable @b : !modelica.variable<100x!modelica.real>

    %t0 = modelica.equation_template inductions = [%i0] attributes {id = "t0"} {
      %0 = modelica.variable_get @a : !modelica.array<100x100x!modelica.real>
      %1 = modelica.variable_get @b : !modelica.array<100x!modelica.real>
      %3 = modelica.subscription %0[%i0] : !modelica.array<100x100x!modelica.real>, index -> !modelica.array<100x!modelica.real>
      %5 = modelica.equation_side %3 : tuple<!modelica.array<100x!modelica.real>>
      %6 = modelica.equation_side %1 : tuple<!modelica.array<100x!modelica.real>>
      modelica.equation_sides %5, %6 : tuple<!modelica.array<100x!modelica.real>>, tuple<!modelica.array<100x!modelica.real>>
    }
    modelica.equation_instance %t0 {
      indices = #modeling<multidim_range [0,99]>, implicit_indices = #modeling<multidim_range [0,99]>
    } : !modelica.equation
 }
}