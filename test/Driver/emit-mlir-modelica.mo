// RUN: marco -emit-mlir-modelica --modelica-frontend=native --model=M -o - %s | FileCheck %s

// CHECK: modelica.model @M

model M
    Real x;
equation
    x = time;
end M;
