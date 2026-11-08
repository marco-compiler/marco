// RUN: marco-opt %s --modelica-instantiate=model=M | FileCheck %s

// Functions are instantiated as 'modelica.function' operations named after
// their dotted path, while the functions implemented by the tool are
// replaced by the corresponding operations. Package constants are replaced
// by their values.
//
// package Math
//   function sin
//     input Real u;
//     output Real y;
//   external "builtin" y = sin(u);
//   end sin;
//   function sq
//     input Real u;
//     input Real k = 2;
//     output Real y;
//   algorithm
//     y := k * u * u;
//   end sq;
//   function norm
//     input Real v[:];
//     output Real result;
//   algorithm
//     result := sqrt(sum(v[i] ^ 2 for i in 1:size(v, 1)));
//   end norm;
//   constant Real pi = 3.14;
// end Math;
// model M
//   Real x = Math.sin(time);
//   Real y = Math.sq(x);
//   Real z = Math.norm({x, y});
//   Real w = Math.pi;
// end M;

// CHECK:       modelica.model @M {
// CHECK-NEXT:      modelica.variable @x : !modelica.real {
// CHECK-NEXT:          %[[time:.*]] = modelica.time
// CHECK-NEXT:          %[[sin:.*]] = modelica.sin %[[time]]
// CHECK-NEXT:          modelica.bind %[[sin]]
// CHECK:           modelica.variable @y : !modelica.real {
// CHECK:               modelica.call @Math.sq
// CHECK:           modelica.variable @z : !modelica.real {
// CHECK:               modelica.call @Math.norm
// CHECK:           modelica.variable @w : !modelica.real {
// CHECK-NEXT:          %[[pi:.*]] = modelica.constant #modelica<real 3.140000e+00>
// CHECK-NEXT:          modelica.bind %[[pi]]

// CHECK:       modelica.function @Math.sq {
// CHECK-NEXT:      modelica.variable input @u : !modelica.real
// CHECK-NEXT:      modelica.variable input @k : !modelica.real {
// CHECK-NEXT:          %[[default:.*]] = modelica.constant #modelica<int 2>
// CHECK-NEXT:          modelica.bind %[[default]]
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.variable output @y : !modelica.real
// CHECK-NEXT:      modelica.algorithm {
// CHECK-NEXT:          modelica.assignment {
// CHECK-NEXT:              modelica.ref @y
// CHECK-NEXT:              modelica.ref @k

// CHECK:       modelica.function @Math.norm {
// CHECK-NEXT:      modelica.variable input @v : tensor<?x!modelica.real>
// CHECK-NEXT:      modelica.variable output @result : !modelica.real

// CHECK-NOT:   modelica.function @Math.sin

module {
  modelica.class package @Math {
    modelica.class function @sin {
      modelica.component input @u : @Real -> !modelica.real
      modelica.component output @y : @Real -> !modelica.real
      modelica.external "builtin" "sin" output {
        %0 = modelica.ref @y : !modelica.real
        %1 = modelica.ref @u : !modelica.real
        modelica.yield %0, %1 : !modelica.real, !modelica.real
      }
    }
    modelica.class function @sq {
      modelica.component input @u : @Real -> !modelica.real
      modelica.component input @k : @Real -> !modelica.real {
        %0 = modelica.constant #modelica<int 2> : !modelica.int
        modelica.bind %0 : !modelica.int
      }
      modelica.component output @y : @Real -> !modelica.real
      modelica.algorithm {
        modelica.assignment {
          %0 = modelica.ref @y : !modelica.real
          %1 = modelica.ref @k : !modelica.real
          %2 = modelica.ref @u : !modelica.real
          %3 = modelica.mul %1, %2 : (!modelica.real, !modelica.real) -> !modelica.real
          %4 = modelica.ref @u : !modelica.real
          %5 = modelica.mul %3, %4 : (!modelica.real, !modelica.real) -> !modelica.real
          modelica.assignment_sides %0 = %5 : !modelica.real = !modelica.real
        }
      }
    }
    modelica.class function @norm {
      modelica.component input @v : @Real -> tensor<?x!modelica.real> dims [:]
      modelica.component output @result : @Real -> !modelica.real
      modelica.algorithm {
        modelica.assignment {
          %0 = modelica.ref @result : !modelica.real
          %1 = modelica.constant #modelica<int 0> : !modelica.int
          %2 = modelica.ref @v : tensor<?x!modelica.real>
          %3 = modelica.size %2, %1 : (tensor<?x!modelica.real>, !modelica.int) -> !modelica.int
          %one = modelica.constant #modelica<int 1> : !modelica.int
          %4 = modelica.sub %3, %one : (!modelica.int, !modelica.int) -> !modelica.int
          %5 = modelica.range %1 to %4 : (!modelica.int, !modelica.int) -> tensor<?x!modelica.int>
          %6 = modelica.reduction sum (%arg0 in %5 : tensor<?x!modelica.int>) -> !modelica.real {
            %8 = modelica.ref @v : tensor<?x!modelica.real>
            %9 = modelica.subscript %8[%arg0] : (tensor<?x!modelica.real>, !modelica.int) -> !modelica.real
            %10 = modelica.constant #modelica<int 2> : !modelica.int
            %11 = modelica.pow %9, %10 : (!modelica.real, !modelica.int) -> !modelica.real
            modelica.yield %11 : !modelica.real
          }
          %7 = modelica.sqrt %6 : (!modelica.real) -> !modelica.real
          modelica.assignment_sides %0 = %7 : !modelica.real = !modelica.real
        }
      }
    }
    modelica.component constant @pi : @Real -> !modelica.real {
      %0 = modelica.constant #modelica<real 3.140000e+00> : !modelica.real
      modelica.bind %0 : !modelica.real
    }
  }
  modelica.class model @M {
    modelica.component @x : @Real -> !modelica.real {
      %0 = modelica.time : !modelica.real
      %1 = modelica.call @Math::@sin(%0) : (!modelica.real) -> !modelica.real
      modelica.bind %1 : !modelica.real
    }
    modelica.component @y : @Real -> !modelica.real {
      %0 = modelica.ref @x : !modelica.real
      %1 = modelica.call @Math::@sq(%0) : (!modelica.real) -> !modelica.real
      modelica.bind %1 : !modelica.real
    }
    modelica.component @z : @Real -> !modelica.real {
      %0 = modelica.ref @x : !modelica.real
      %1 = modelica.ref @y : !modelica.real
      %2 = modelica.array %0, %1 : (!modelica.real, !modelica.real) -> tensor<2x!modelica.real>
      %3 = modelica.call @Math::@norm(%2) : (tensor<2x!modelica.real>) -> !modelica.real
      modelica.bind %3 : !modelica.real
    }
    modelica.component @w : @Real -> !modelica.real {
      %0 = modelica.ref @Math::@pi : !modelica.real
      modelica.bind %0 : !modelica.real
    }
  }
}
