// RUN: marco -mc1 -S --omc-bypass -o - %s | FileCheck %s

// CHECK: text

modelica.model @M {

}
