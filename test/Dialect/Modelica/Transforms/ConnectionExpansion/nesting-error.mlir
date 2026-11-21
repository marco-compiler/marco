// RUN: marco-opt %s --modelica-connection-expansion --verify-diagnostics

// Connect-equations can't be nested in operations other than for-equations
// after the instantiation.

module {
  modelica.model @NestingError {
    modelica.variable parameter @b : !modelica.bool
    modelica.instance @a : !modelica.object<@C> {
      modelica.instance connector @p : !modelica.object<@Pin> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
    }
    modelica.instance @c : !modelica.object<@C> {
      modelica.instance connector @p : !modelica.object<@Pin> {
        modelica.variable @v : !modelica.real
        modelica.variable flow @i : !modelica.real
      }
    }
    modelica.equations {
      modelica.if_equation {
        %0 = modelica.ref @b : !modelica.bool
        modelica.yield %0 : !modelica.bool
      } then {
        // expected-error @+1 {{connect-equations can only be nested in for-equations after the instantiation}}
        modelica.connect {
          %0 = modelica.ref @a : !modelica.object<@C>
          %1 = modelica.member %0, @p : (!modelica.object<@C>) -> !modelica.object<@Pin>
          %2 = modelica.ref @c : !modelica.object<@C>
          %3 = modelica.member %2, @p : (!modelica.object<@C>) -> !modelica.object<@Pin>
          modelica.yield %1, %3 : !modelica.object<@Pin>, !modelica.object<@Pin>
        }
      }
    }
  }
}
