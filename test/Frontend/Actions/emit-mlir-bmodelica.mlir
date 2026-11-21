// RUN: marco -mc1 -emit-mlir-bmodelica -o - %s | FileCheck %s

// CHECK: bmodelica.model

bmodelica.model @M {

}
