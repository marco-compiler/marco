// RUN: marco-opt %s --modelica-instantiate=model=M | FileCheck %s

// Inheritance and merging of modifications: the modifications of the
// extends-clause override the ones of the base class, which override the
// ones of the short class definitions.
//
// type Voltage = Real(unit = "V", nominal = 10);
// model Base
//   parameter Real k = 1;
//   Voltage v(start = 2);
//   Real x;
// equation
//   x = k * v;
// end Base;
// model M
//   extends Base(k = 2, v(nominal = 5));
//   Real y[2](each start = 3);
// equation
//   y = {x, v};
// end M;

// CHECK-NOT: modelica.class
// CHECK:       modelica.model @M {
// CHECK:           modelica.variable parameter @k : !modelica.real {
// CHECK-NEXT:          %[[k:.*]] = modelica.constant #modelica<int 2>
// CHECK-NEXT:          modelica.bind %[[k]]
// CHECK:           modelica.variable @v : !modelica.real {
// CHECK-NEXT:          modelica.modify @nominal {
// CHECK-NEXT:              %[[nominal:.*]] = modelica.constant #modelica<int 5>
// CHECK-NEXT:              modelica.bind %[[nominal]]
// CHECK:               modelica.modify @start {
// CHECK-NEXT:              %[[start:.*]] = modelica.constant #modelica<int 2>
// CHECK-NEXT:              modelica.bind %[[start]]
// CHECK:               modelica.modify @unit {
// CHECK-NEXT:              %[[unit:.*]] = modelica.constant #modelica<string "V">
// CHECK-NEXT:              modelica.bind %[[unit]]
// CHECK:           modelica.variable @x : !modelica.real
// CHECK:           modelica.variable @y : tensor<2x!modelica.real> {
// CHECK-NEXT:          modelica.modify each @start {
// CHECK:           modelica.equations {
// CHECK-NEXT:          modelica.equation {
// CHECK-NEXT:              %[[x:.*]] = modelica.ref @x : !modelica.real
// CHECK-NEXT:              %[[k:.*]] = modelica.ref @k : !modelica.real
// CHECK-NEXT:              %[[v:.*]] = modelica.ref @v : !modelica.real
// CHECK-NEXT:              %[[mul:.*]] = modelica.mul %[[k]], %[[v]]
// CHECK-NEXT:              modelica.equation_sides %[[x]] = %[[mul]]
// CHECK:               modelica.equation {
// CHECK:                   modelica.ref @y : tensor<2x!modelica.real>

module {
  modelica.class short type @Voltage {
    modelica.extends @Real {
      modelica.modify @unit {
        %0 = modelica.constant #modelica<string "V"> : !modelica.string
        modelica.bind %0 : !modelica.string
      }
      modelica.modify @nominal {
        %0 = modelica.constant #modelica<int 10> : !modelica.int
        modelica.bind %0 : !modelica.int
      }
    }
  }
  modelica.class model @Base {
    modelica.component parameter @k : @Real -> !modelica.real {
      %0 = modelica.constant #modelica<int 1> : !modelica.int
      modelica.bind %0 : !modelica.int
    }
    modelica.component @v : @Voltage -> !modelica.real {
      modelica.modify @start {
        %0 = modelica.constant #modelica<int 2> : !modelica.int
        modelica.bind %0 : !modelica.int
      }
    }
    modelica.component @x : @Real -> !modelica.real
    modelica.equations {
      modelica.equation {
        %0 = modelica.ref @x : !modelica.real
        %1 = modelica.ref @k : !modelica.real
        %2 = modelica.ref @v : !modelica.real
        %3 = modelica.mul %1, %2 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.equation_sides %0 = %3 : !modelica.real = !modelica.real
      }
    }
  }
  modelica.class model @M {
    modelica.extends @Base {
      modelica.modify @k {
        %0 = modelica.constant #modelica<int 2> : !modelica.int
        modelica.bind %0 : !modelica.int
      }
      modelica.modify @v {
        modelica.modify @nominal {
          %0 = modelica.constant #modelica<int 5> : !modelica.int
          modelica.bind %0 : !modelica.int
        }
      }
    }
    modelica.component @y : @Real -> tensor<2x!modelica.real> {
      modelica.modify each @start {
        %0 = modelica.constant #modelica<int 3> : !modelica.int
        modelica.bind %0 : !modelica.int
      }
    } dims [2]
    modelica.equations {
      modelica.equation {
        %0 = modelica.ref @y : tensor<2x!modelica.real>
        %1 = modelica.ref @x : !modelica.real
        %2 = modelica.ref @v : !modelica.real
        %3 = modelica.array %1, %2 : (!modelica.real, !modelica.real) -> tensor<2x!modelica.real>
        modelica.equation_sides %0 = %3 : tensor<2x!modelica.real> = tensor<2x!modelica.real>
      }
    }
  }
}
