// RUN: marco -emit-mlir-bmodelica -o - %s | FileCheck %s

// CHECK: bmodelica.model

bmodelica.model @M {

}
