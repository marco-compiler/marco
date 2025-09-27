// RUN: marco -emit-ast --omc-bypass -o - %s | FileCheck %s

// CHECK: "kind": "root"

model M
end M;
