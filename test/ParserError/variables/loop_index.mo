// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion. Unknown variable identifier id1.

model AccessesDependingOnIndices
    Real[3, 4] x;
    Real variable;
equation
    for idx1 in 1:3 loop
        for idx2 in 1:4 loop
            der(x[id1, idx2]) = 2 * der(x[2, 2]) - 4;
        end for;
    end for;
end AccessesDependingOnIndices;
