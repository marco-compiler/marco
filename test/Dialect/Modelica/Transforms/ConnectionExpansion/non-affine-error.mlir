// RUN: marco-opt %s --modelica-connection-expansion --verify-diagnostics

// Connect-equations with non-affine indices are not supported yet.
//
// for k in 1:2 loop
//   connect(r[k * k].n, r[k].p);
// end for;

module {
  modelica.model @NonAffineError {
    modelica.instance @r : tensor<4x!modelica.object<@TwoPin>> {
      modelica.instance connector @p : !modelica.object<@Pin> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
      modelica.instance connector @n : !modelica.object<@Pin> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
    }
    modelica.equations {
      modelica.for_equation %arg0 : !modelica.int in {
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        %1 = modelica.constant #modelica<int 2> : !modelica.int
        %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<2x!modelica.int>
        modelica.yield %2 : tensor<2x!modelica.int>
      } do {
        // expected-error @+1 {{non-affine indices in connect-equations are not supported yet}}
        modelica.connect {
          %0 = modelica.ref @r : tensor<4x!modelica.object<@TwoPin>>
          %1 = modelica.mul %arg0, %arg0 : (!modelica.int, !modelica.int) -> !modelica.int
          %2 = modelica.subscript %0[%1] : (tensor<4x!modelica.object<@TwoPin>>, !modelica.int) -> !modelica.object<@TwoPin>
          %3 = modelica.member %2, @n : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
          %4 = modelica.ref @r : tensor<4x!modelica.object<@TwoPin>>
          %5 = modelica.subscript %4[%arg0] : (tensor<4x!modelica.object<@TwoPin>>, !modelica.int) -> !modelica.object<@TwoPin>
          %6 = modelica.member %5, @p : (!modelica.object<@TwoPin>) -> !modelica.object<@Pin>
          modelica.yield %3, %6 : !modelica.object<@Pin>, !modelica.object<@Pin>
        }
      }
    }
  }
}
