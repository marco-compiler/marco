// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Unknown type or class identifier Model1.

package A
    model Model1
        parameter Integer par;
    end Model1;
end A;

package B
    record Model2
        Real x;
    end Model2;
    
    function foo
        input Model1 m;
        output Integer y;
    algorithm
        y := m.par;
    end foo;
end B;