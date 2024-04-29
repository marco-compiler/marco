// RUN: marco -S --omc-bypass -o - %s | FileCheck %s

// CHECK: text

modelica.model @M {

}
