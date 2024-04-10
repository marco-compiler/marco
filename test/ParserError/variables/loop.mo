// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion. Unknown variable identifier variale at line 11, column 47. Did you mean variable?

model AccessesDependingOnIndices
    Real[3, 4] x;
    Real variable;
equation
    for i in 1:3 loop
        for j in 1:4 loop
            der(x[i, j]) = 2 * der(x[2, 2]) - variale;
        end for;
    end for;
end AccessesDependingOnIndices;
