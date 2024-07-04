// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Unknown function identifier foo1.

package A
    function foo1
        input Integer[:] x;
        output Integer[:,:] y;
    algorithm
        y := diagonal(x);
    end foo1;
end A;

package B
    function foo2
        input Integer[:] x;
        output Integer[:,:] y;
    algorithm
        y := diagonal(x);
    end foo2;

    function bar
        input Integer[:] x;
        output Integer[:,:] y;
    algorithm
        y := foo1(x);
    end bar;
end B;

