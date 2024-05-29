// RUN: marco %s -o - --model=Test -Xomc=-d=nonfScalarize -emit-base-modelica | FileCheck %s

// CHECK:       model 'Test'
// CHECK-NEXT:      Real[10] 'x'(each fixed = true, each start = 0.0);
// CHECK-NEXT:  equation
// CHECK-NEXT:      for 'i' in 1:10 loop
// CHECK-NEXT:          'x'['i'] = 1.0 - der('x'['i']);
// CHECK-NEXT:      end for;
// CHECK-NEXT:  end 'Test';

model Test
    Real[10] x(each fixed = true, each start = 0);
equation
    for i in 1:10 loop
        x[i] = 1 - der(x[i]);
    end for;
end Test;
