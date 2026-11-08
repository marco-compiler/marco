// RUN: marco-opt %s --modelica-instantiate=model=M | FileCheck %s

// Redeclarations of packages, functions and components.
// The constants of the redeclared package are used for the dimensions, and
// the function redeclared by the class-extends redeclaration is
// instantiated once.
//
// partial package PartialMedium
//   constant Integer nX;
//   replaceable partial function h
//     input Real T;
//     output Real y;
//   end h;
// end PartialMedium;
// package Water
//   extends PartialMedium(nX = 2);
//   redeclare function extends h
//   algorithm
//     y := 4184 * T;
//   end h;
// end Water;
// model Volume
//   replaceable package Medium = PartialMedium;
//   Real X[Medium.nX];
//   Real e = Medium.h(X[1]);
// end Volume;
// model M
//   Volume v1(redeclare package Medium = Water);
//   Volume v2(redeclare package Medium = Water);
//   replaceable Real r = 1;
//   M2 m2(redeclare Integer n = 3);
//   model M2
//     replaceable Real n = 1;
//   end M2;
// end M;

// CHECK:       modelica.model @M {
// CHECK:           modelica.instance model @v1 : !modelica.object<@Volume> {
// CHECK-NEXT:          modelica.variable @X : tensor<2x!modelica.real>
// CHECK:               modelica.call @Water.h
// CHECK:           modelica.instance model @v2 : !modelica.object<@Volume> {
// CHECK-NEXT:          modelica.variable @X : tensor<2x!modelica.real>
// CHECK:               modelica.call @Water.h
// CHECK:           modelica.instance model @m2 : !modelica.object<@M::@M2> {
// CHECK-NEXT:          modelica.variable @n : !modelica.int {
// CHECK-NEXT:              %[[n:.*]] = modelica.constant #modelica<int 3>
// CHECK-NEXT:              modelica.bind %[[n]]

// CHECK:       modelica.function @Water.h {
// CHECK-NEXT:      modelica.variable input @T : !modelica.real
// CHECK-NEXT:      modelica.variable output @y : !modelica.real
// CHECK-NEXT:      modelica.algorithm {
// CHECK-NOT:   modelica.function

module {
  modelica.class partial package @PartialMedium {
    modelica.component constant @nX : @Integer -> !modelica.int
    modelica.class replaceable partial function @h {
      modelica.component input @T : @Real -> !modelica.real
      modelica.component output @y : @Real -> !modelica.real
    }
  }
  modelica.class package @Water {
    modelica.extends @PartialMedium {
      modelica.modify @nX {
        %0 = modelica.constant #modelica<int 2> : !modelica.int
        modelica.bind %0 : !modelica.int
      }
    }
    modelica.class redeclare function extends @h {
      modelica.extends inherited
      modelica.algorithm {
        modelica.assignment {
          %0 = modelica.ref @y : !modelica.real
          %1 = modelica.constant #modelica<int 4184> : !modelica.int
          %2 = modelica.ref @T : !modelica.real
          %3 = modelica.mul %1, %2 : (!modelica.int, !modelica.real) -> !modelica.real
          modelica.assignment_sides %0 = %3 : !modelica.real = !modelica.real
        }
      }
    }
  }
  modelica.class model @Volume {
    modelica.class replaceable short package @Medium {
      modelica.extends @PartialMedium
    }
    modelica.component @X : @Real -> tensor<?x!modelica.real> dims [?] {
      %0 = modelica.ref @Medium::@nX : !modelica.int
      modelica.yield %0 : !modelica.int
    }
    modelica.component @e : @Real -> !modelica.real {
      %0 = modelica.ref @X : tensor<?x!modelica.real>
      %1 = modelica.constant #modelica<int 1> : !modelica.int
      %2 = modelica.subscript %0[%1] : (tensor<?x!modelica.real>, !modelica.int) -> !modelica.real
      %3 = modelica.call @Medium::@h(%2) : (!modelica.real) -> !modelica.real
      modelica.bind %3 : !modelica.real
    }
  }
  modelica.class model @M {
    modelica.component @v1 : @Volume -> !modelica.object<@Volume> {
      modelica.class redeclare short package @Medium {
        modelica.extends @Water
      }
    }
    modelica.component @v2 : @Volume -> !modelica.object<@Volume> {
      modelica.class redeclare short package @Medium {
        modelica.extends @Water
      }
    }
    modelica.component replaceable @r : @Real -> !modelica.real {
      %0 = modelica.constant #modelica<int 1> : !modelica.int
      modelica.bind %0 : !modelica.int
    }
    modelica.component @m2 : @M2 -> !modelica.object<@M::@M2> {
      modelica.component redeclare @n : @Integer -> !modelica.int {
        %0 = modelica.constant #modelica<int 3> : !modelica.int
        modelica.bind %0 : !modelica.int
      }
    }
    modelica.class model @M2 {
      modelica.component replaceable @n : @Real -> !modelica.real {
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        modelica.bind %0 : !modelica.int
      }
    }
  }
}
