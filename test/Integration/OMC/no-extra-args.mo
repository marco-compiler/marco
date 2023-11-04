// RUN: marco %s -o - --model=Test -emit-base-modelica | FileCheck %s

// CHECK:       model 'Test'
// CHECK-NEXT:      Real 'x'(fixed = true, start = 0.0);
// CHECK-NEXT:  equation
// CHECK-NEXT:      'x' = 1.0 - der('x');
// CHECK-NEXT:  end 'Test';

model Test
    Real x(fixed = true, start = 0);
equation
    x = 1 - der(x);
end Test;
