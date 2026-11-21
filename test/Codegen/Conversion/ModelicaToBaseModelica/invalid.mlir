// RUN: marco-opt %s --split-input-file --verify-diagnostics --convert-modelica-to-bmodelica

// Constructs without a counterpart in Base Modelica are reported.

modelica.model @WhenEquation {
  modelica.variable @x : !modelica.real
  modelica.equations {
    // expected-error @+1 {{when-equations are not supported yet}}
    modelica.when_equation {
      %0 = modelica.initial : !modelica.bool
      modelica.yield %0 : !modelica.bool
    } then {
    }
  }
}

// -----

// String variables are dropped, and their usages are reported.

modelica.model @StringVariable {
  modelica.variable parameter @s : !modelica.string
  modelica.variable @n : !modelica.int
  modelica.equations {
    modelica.equation {
      %n = modelica.ref @n : !modelica.int
      // expected-error @+1 {{String values are not supported}}
      %s = modelica.ref @s : !modelica.string
      %0 = modelica.call @Strings.length(%s) : (!modelica.string) -> !modelica.int
      modelica.equation_sides %n = %0 : !modelica.int = !modelica.int
    }
  }
}

// -----

modelica.model @UnbalancedIfEquation {
  modelica.variable @x : !modelica.real
  modelica.equations {
    // expected-error @+1 {{the branches of an if-equation whose condition is not known at compile time must have the same number of equations}}
    modelica.if_equation {
      %0 = modelica.time : !modelica.real
      %1 = modelica.constant #modelica<int 1> : !modelica.int
      %2 = modelica.gt %0, %1 : (!modelica.real, !modelica.int) -> !modelica.bool
      modelica.yield %2 : !modelica.bool
    } then {
      modelica.equation {
        %0 = modelica.ref @x : !modelica.real
        %1 = modelica.constant #modelica<int 1> : !modelica.int
        modelica.equation_sides %0 = %1 : !modelica.real = !modelica.int
      }
    }
  }
}

// -----

modelica.model @NonConstantRange {
  modelica.variable @n : !modelica.int
  modelica.variable @x : tensor<3x!modelica.real>
  modelica.equations {
    // expected-error @+1 {{the iteration range of the for-equation must be known at compile time}}
    modelica.for_equation %i : !modelica.int in {
      %0 = modelica.constant #modelica<int 1> : !modelica.int
      %n = modelica.ref @n : !modelica.int
      %1 = modelica.range %0 to %n : (!modelica.int, !modelica.int) -> tensor<?x!modelica.int>
      modelica.yield %1 : tensor<?x!modelica.int>
    } do {
    }
  }
}
