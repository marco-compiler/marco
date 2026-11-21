// RUN: marco -mc1 -emit-mlir-modelica -o - %s | FileCheck %s

// The native frontend emits the classes in the Modelica dialect, right after
// their lowering. The model is neither instantiated nor flattened.

// CHECK-DAG:   @Pin
// CHECK-DAG:   @Component
// CHECK:       modelica.model @M {
// CHECK-DAG:       modelica.component @a : @Component
// CHECK-DAG:       modelica.component @b : @Component
// CHECK:           modelica.connect
// CHECK-NOT:   modelica.instance

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
