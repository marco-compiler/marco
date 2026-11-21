// RUN: marco-opt %s --modelica-connection-expansion | FileCheck %s

// Slices whose bounds point against the step, by less than the step, are
// empty and connect nothing. Their size is not known statically.
//
// model EmptySlice
//   Pin a[6], b[6];
// equation
//   connect(a[5:2:4], b[1:2:0]);
//   connect(a[2:-2:3], b[1:-2:2]);
// end EmptySlice;

// CHECK:       modelica.model @EmptySlice
// CHECK-NOT:       modelica.connect
// CHECK:           modelica.equations {
// CHECK-NEXT:      }
// CHECK-NEXT:  }

module {
  modelica.model @EmptySlice {
    modelica.instance connector @a : tensor<6x!modelica.object<@Pin>> {
      modelica.variable @v : !modelica.real
    }
    modelica.instance connector @b : tensor<6x!modelica.object<@Pin>> {
      modelica.variable @v : !modelica.real
    }
    modelica.equations {
      modelica.connect {
        %0 = modelica.ref @a : tensor<6x!modelica.object<@Pin>>
        %1 = modelica.constant #modelica<int 4> : !modelica.int
        %2 = modelica.constant #modelica<int 3> : !modelica.int
        %3 = modelica.constant #modelica<int 2> : !modelica.int
        %4 = modelica.range %1 to %2 step %3 : (!modelica.int, !modelica.int, !modelica.int) -> tensor<?x!modelica.int>
        %5 = modelica.subscript %0[%4] : (tensor<6x!modelica.object<@Pin>>, tensor<?x!modelica.int>) -> tensor<?x!modelica.object<@Pin>>
        %6 = modelica.ref @b : tensor<6x!modelica.object<@Pin>>
        %7 = modelica.constant #modelica<int 0> : !modelica.int
        %8 = modelica.constant #modelica<int -1> : !modelica.int
        %9 = modelica.range %7 to %8 step %3 : (!modelica.int, !modelica.int, !modelica.int) -> tensor<?x!modelica.int>
        %10 = modelica.subscript %6[%9] : (tensor<6x!modelica.object<@Pin>>, tensor<?x!modelica.int>) -> tensor<?x!modelica.object<@Pin>>
        modelica.yield %5, %10 : tensor<?x!modelica.object<@Pin>>, tensor<?x!modelica.object<@Pin>>
      }
      modelica.connect {
        %0 = modelica.ref @a : tensor<6x!modelica.object<@Pin>>
        %1 = modelica.constant #modelica<int 1> : !modelica.int
        %2 = modelica.constant #modelica<int 2> : !modelica.int
        %3 = modelica.constant #modelica<int -2> : !modelica.int
        %4 = modelica.range %1 to %2 step %3 : (!modelica.int, !modelica.int, !modelica.int) -> tensor<?x!modelica.int>
        %5 = modelica.subscript %0[%4] : (tensor<6x!modelica.object<@Pin>>, tensor<?x!modelica.int>) -> tensor<?x!modelica.object<@Pin>>
        %6 = modelica.ref @b : tensor<6x!modelica.object<@Pin>>
        %7 = modelica.constant #modelica<int 0> : !modelica.int
        %8 = modelica.constant #modelica<int 1> : !modelica.int
        %9 = modelica.range %7 to %8 step %3 : (!modelica.int, !modelica.int, !modelica.int) -> tensor<?x!modelica.int>
        %10 = modelica.subscript %6[%9] : (tensor<6x!modelica.object<@Pin>>, tensor<?x!modelica.int>) -> tensor<?x!modelica.object<@Pin>>
        modelica.yield %5, %10 : tensor<?x!modelica.object<@Pin>>, tensor<?x!modelica.object<@Pin>>
      }
    }
  }
}
