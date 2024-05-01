// RUN: marco -mc1 -emit-llvm --omc-bypass -o - %s | FileCheck %s

// CHECK: @modelName = internal constant [2 x i8] c"M\00

bmodelica.model @M {

}
