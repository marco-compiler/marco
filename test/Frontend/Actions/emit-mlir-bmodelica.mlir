// RUN: marco -mc1 -emit-mlir-bmodelica --omc-bypass -o - %s | FileCheck %s

// CHECK: bmodelica.model

bmodelica.model @M {

}
