// RUN: marco -emit-llvm -S --omc-bypass -o - %s | FileCheck %s

// CHECK: @modelName = internal constant [2 x i8] c"M\00

modelica.model @M {

}
