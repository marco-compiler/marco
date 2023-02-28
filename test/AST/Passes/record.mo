// RUN: marco %s --omc-bypass --emit-mlir -o %t
// RUN: cat %t | FileCheck %s

// only reals member must be present
// CHECK-LABEL: @A
// CHECK: !modelica.member<{{([0-9]+x)?}}!modelica.real>
// CHECK-NOT: !modelica.member<{{([0-9]+x)?}}!modelica.{{[^r]}}

// the equations must be decomposed to use the real members
// CHECK-LABEL modelica.equation
// CHECK: modelica.equation_sides %{{[0-9]+}}, %{{[0-9]+}} : tuple<!modelica.real>, tuple<!modelica.real>

model 'A'
function 'Complex.\'*\'.multiply' "Multiply two complex numbers"
  input 'Complex' 'c1';
  input 'Complex' 'c2';
  output 'Complex' 'c3';
algorithm
  'c3' := 'Complex.\'constructor\'.fromReal'('c1'.'re' * 'c2'.'re' - 'c1'.'im' * 'c2'.'im', 'c1'.'re' * 'c2'.'im' + 'c1'.'im' * 'c2'.'re');
  annotation(Inline = true);
end 'Complex.\'*\'.multiply';

function 'Complex.\'constructor\'.fromReal'
  input Real 're';
  input Real 'im' = 0.0;
  output 'Complex' 'result';
algorithm
  annotation(Inline = true);
end 'Complex.\'constructor\'.fromReal';

record 'Complex'
  Real 're';
  Real 'im';
end 'Complex';

  'Complex'[2] 'r';
  'Complex' 'y';
equation
  'y' = 'Complex.\'*\'.multiply'('r'[1], 'Complex.\'constructor\'.fromReal'('r.re'[1], 'r.im'[2]));
  'y.re' = 'r.im'[1];
  for i in 1:2 loop
    'r.re'[i] = 2.0 * 'y.im';
    'r.im'[i] = 2.0 * 'y.re';
  end for;
end 'A';
