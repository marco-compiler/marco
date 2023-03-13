// RUN: marco %s --omc-bypass --emit-mlir -o %t
// RUN: cat %t | FileCheck %s

// CHECK-LABEL: @SD
// CHECK: @"c.x[1]" : !modelica.variable<3x!modelica.real>
// CHECK: @"c.x[2]" : !modelica.variable<4x!modelica.real>
// CHECK: @"c.x[3]" : !modelica.variable<5x!modelica.real>

model 'SD'
  parameter Integer 'N' = 3;
  parameter Real[3] 'p' = {1.0, 1.5, 2.0};
  Real[3] 'c.c.f';
  Real[3] 'c.c.e';
  Real[3, {3, 4, 5}] 'c.x';
  parameter Real[3] 'c.p' = 'p'[:];
  parameter Integer[3] 'c.N' = {3, 4, 5};
  parameter Real 's.p' = 3.0;
  Real 's.c.e';
  Real 's.c.f';
equation
  'c.x'[:,1] = 'c.c.e'[:];
  'c.x'[:,'c.N'[:]] = 'c.c.f'[:];

  for '$i1' in 1:3 loop
    for 'i' in 2:{3, 4, 5}['$i1'] loop
      'c.x'['$i1','i'] = 'c.x'['$i1','i' - 1] + 'c.p'['$i1'];
    end for;
  end for;

  's.c.f' = 's.p';
  'c.c.e'[2] = 'c.c.e'[1];
  's.c.e' = 'c.c.e'[1];
  'c.c.e'[3] = 'c.c.e'[1];
  'c.c.f'[3] + 's.c.f' + 'c.c.f'[2] + 'c.c.f'[1] = 0.0;
end 'SD';