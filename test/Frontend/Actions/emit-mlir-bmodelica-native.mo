// RUN: marco -mc1 -emit-mlir-bmodelica --modelica-frontend=native --model=M -o - %s | FileCheck %s

// The models flattened by the native frontend are converted to Base Modelica.

// CHECK:       bmodelica.model @M {
// CHECK-DAG:       bmodelica.variable @a.p.v : !bmodelica.variable<!bmodelica.real>
// CHECK-DAG:       bmodelica.variable @a.p.i : !bmodelica.variable<!bmodelica.real>
// CHECK-DAG:       bmodelica.variable @b.p.v : !bmodelica.variable<!bmodelica.real>
// CHECK-DAG:       bmodelica.variable @b.p.i : !bmodelica.variable<!bmodelica.real>
// CHECK:           bmodelica.dynamic {
// CHECK:               bmodelica.variable.get @a.p.v
// CHECK:               bmodelica.time

connector Pin
    Real v;
    flow Real i;
end Pin;

model Component
    Pin p;
end Component;

model M
    Component a, b;
equation
    connect(a.p, b.p);
    a.p.v = time;
end M;
