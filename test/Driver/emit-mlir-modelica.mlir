// RUN: marco -emit-mlir-modelica --omc-bypass -o - %s | FileCheck %s

// CHECK: bmodelica.model

bmodelica.model @M {

}
