// RUN: marco -emit-ast --omc-bypass -o - %s | FileCheck %s

// CHECK: "node_type": "root"

model M
end M;
