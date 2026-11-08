// RUN: marco -mc1 -emit-mlir-modelica --modelica-frontend=native --model=M -o - %s | FileCheck %s

// The native frontend emits the flat model in the Modelica dialect.

// CHECK:       modelica.model @M {
// CHECK-DAG:       modelica.variable @a.p.v : !modelica.real
// CHECK-DAG:       modelica.variable @a.p.i : !modelica.real
// CHECK-DAG:       modelica.variable @b.p.v : !modelica.real
// CHECK-DAG:       modelica.variable @b.p.i : !modelica.real
// CHECK-NOT:       modelica.instance
// CHECK-NOT:       modelica.connect

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
