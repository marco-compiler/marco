// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: size: expected between 1 and 2 argument(s) but got 0.

function sizeArray
    input Real[:,:] x;
    output Integer[2] y;
algorithm
    y := size();
end sizeArray;