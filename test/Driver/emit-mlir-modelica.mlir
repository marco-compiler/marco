// RUN: marco -emit-mlir-modelica --omc-bypass -o - %s | FileCheck %s

// CHECK: modelica.model

modelica.model @M {

}
