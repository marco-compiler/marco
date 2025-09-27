// RUN: marco -mc1 -emit-ast --omc-bypass -o - %s | FileCheck %s

// CHECK: "kind": "root"

model M
end M;
