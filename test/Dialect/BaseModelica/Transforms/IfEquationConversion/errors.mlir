// RUN: marco-opt %s --split-input-file --convert-if-equations --verify-diagnostics

// COM: Branches writing to different variables must be rejected.

bmodelica.model @DifferentLHS {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    bmodelica.dynamic {
        // expected-error @below {{all branches of an if_equation must write to the same left-hand side}}
        bmodelica.if_equation if {
            %cond = bmodelica.constant #bmodelica<bool true> : !bmodelica.bool
            bmodelica.yield %cond : !bmodelica.bool
        } then {
            bmodelica.equation {
                %x   = bmodelica.variable.get @x : !bmodelica.int
                %one = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
                %lhs = bmodelica.equation_side %x   : tuple<!bmodelica.int>
                %rhs = bmodelica.equation_side %one : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        } else {
            bmodelica.equation {
                %y    = bmodelica.variable.get @y : !bmodelica.int
                %zero = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
                %lhs  = bmodelica.equation_side %y    : tuple<!bmodelica.int>
                %rhs  = bmodelica.equation_side %zero : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        }
    }
}

// -----

// COM: More than one equation in the then-branch must be rejected.

bmodelica.model @MultipleEquationsInThen {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    bmodelica.dynamic {
        // expected-error @below {{if_equation then-branch must contain exactly one equation}}
        bmodelica.if_equation if {
            %cond = bmodelica.constant #bmodelica<bool true> : !bmodelica.bool
            bmodelica.yield %cond : !bmodelica.bool
        } then {
            bmodelica.equation {
                %x   = bmodelica.variable.get @x : !bmodelica.int
                %one = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
                %lhs = bmodelica.equation_side %x   : tuple<!bmodelica.int>
                %rhs = bmodelica.equation_side %one : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
            bmodelica.equation {
                %y   = bmodelica.variable.get @y : !bmodelica.int
                %one = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
                %lhs = bmodelica.equation_side %y   : tuple<!bmodelica.int>
                %rhs = bmodelica.equation_side %one : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        } else {
            bmodelica.equation {
                %x    = bmodelica.variable.get @x : !bmodelica.int
                %zero = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
                %lhs  = bmodelica.equation_side %x    : tuple<!bmodelica.int>
                %rhs  = bmodelica.equation_side %zero : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        }
    }
}

// -----

// COM: A single non-EquationOp in the then-branch must be rejected.
// COM: The natural case is a nested if_equation (else-if chain).

bmodelica.model @NonEquationOpInThen {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    bmodelica.dynamic {
        // expected-error @below {{if_equation then-branch must contain an equation op}}
        bmodelica.if_equation if {
            %cond = bmodelica.constant #bmodelica<bool true> : !bmodelica.bool
            bmodelica.yield %cond : !bmodelica.bool
        } then {
            bmodelica.if_equation if {
                %c2 = bmodelica.constant #bmodelica<bool false> : !bmodelica.bool
                bmodelica.yield %c2 : !bmodelica.bool
            } then {
                bmodelica.equation {
                    %x   = bmodelica.variable.get @x : !bmodelica.int
                    %one = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
                    %lhs = bmodelica.equation_side %x   : tuple<!bmodelica.int>
                    %rhs = bmodelica.equation_side %one : tuple<!bmodelica.int>
                    bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
                }
            } else {
                bmodelica.equation {
                    %x    = bmodelica.variable.get @x : !bmodelica.int
                    %zero = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
                    %lhs  = bmodelica.equation_side %x    : tuple<!bmodelica.int>
                    %rhs  = bmodelica.equation_side %zero : tuple<!bmodelica.int>
                    bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
                }
            }
        } else {
            bmodelica.equation {
                %x    = bmodelica.variable.get @x : !bmodelica.int
                %zero = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
                %lhs  = bmodelica.equation_side %x    : tuple<!bmodelica.int>
                %rhs  = bmodelica.equation_side %zero : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        }
    }
}

// -----

// COM: More than one equation in the else-branch must be rejected.

bmodelica.model @MultipleEquationsInElse {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    bmodelica.dynamic {
        // expected-error @below {{if_equation else-branch must contain exactly one equation}}
        bmodelica.if_equation if {
            %cond = bmodelica.constant #bmodelica<bool true> : !bmodelica.bool
            bmodelica.yield %cond : !bmodelica.bool
        } then {
            bmodelica.equation {
                %x   = bmodelica.variable.get @x : !bmodelica.int
                %one = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
                %lhs = bmodelica.equation_side %x   : tuple<!bmodelica.int>
                %rhs = bmodelica.equation_side %one : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        } else {
            bmodelica.equation {
                %x    = bmodelica.variable.get @x : !bmodelica.int
                %zero = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
                %lhs  = bmodelica.equation_side %x    : tuple<!bmodelica.int>
                %rhs  = bmodelica.equation_side %zero : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
            bmodelica.equation {
                %y    = bmodelica.variable.get @y : !bmodelica.int
                %zero = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
                %lhs  = bmodelica.equation_side %y    : tuple<!bmodelica.int>
                %rhs  = bmodelica.equation_side %zero : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        }
    }
}

// -----

// COM: A single non-EquationOp in the else-branch must be rejected.
// COM: The natural case is a nested if_equation forming an else-if chain.

bmodelica.model @ElseIfChain {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    bmodelica.dynamic {
        // expected-error @below {{if_equation else-branch must contain an equation op}}
        bmodelica.if_equation if {
            %cond = bmodelica.constant #bmodelica<bool true> : !bmodelica.bool
            bmodelica.yield %cond : !bmodelica.bool
        } then {
            bmodelica.equation {
                %x   = bmodelica.variable.get @x : !bmodelica.int
                %one = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
                %lhs = bmodelica.equation_side %x   : tuple<!bmodelica.int>
                %rhs = bmodelica.equation_side %one : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        } else {
            bmodelica.if_equation if {
                %c2 = bmodelica.constant #bmodelica<bool false> : !bmodelica.bool
                bmodelica.yield %c2 : !bmodelica.bool
            } then {
                bmodelica.equation {
                    %x   = bmodelica.variable.get @x : !bmodelica.int
                    %two = bmodelica.constant #bmodelica<int 2> : !bmodelica.int
                    %lhs = bmodelica.equation_side %x   : tuple<!bmodelica.int>
                    %rhs = bmodelica.equation_side %two : tuple<!bmodelica.int>
                    bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
                }
            } else {
                bmodelica.equation {
                    %x    = bmodelica.variable.get @x : !bmodelica.int
                    %zero = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
                    %lhs  = bmodelica.equation_side %x    : tuple<!bmodelica.int>
                    %rhs  = bmodelica.equation_side %zero : tuple<!bmodelica.int>
                    bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
                }
            }
        }
    }
}

// -----

// COM: An equation whose LHS side in the else-branch carries more than one
// COM: value must be rejected (then-branch is well-formed with exactly one).

bmodelica.model @MultipleLHSValuesInElseOnly {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    bmodelica.dynamic {
        // expected-error @below {{expected exactly one LHS value in each branch equation}}
        bmodelica.if_equation if {
            %cond = bmodelica.constant #bmodelica<bool true> : !bmodelica.bool
            bmodelica.yield %cond : !bmodelica.bool
        } then {
            bmodelica.equation {
                %x   = bmodelica.variable.get @x : !bmodelica.int
                %one = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
                %lhs = bmodelica.equation_side %x   : tuple<!bmodelica.int>
                %rhs = bmodelica.equation_side %one : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        } else {
            bmodelica.equation {
                %x   = bmodelica.variable.get @x : !bmodelica.int
                %y   = bmodelica.variable.get @y : !bmodelica.int
                %lhs = bmodelica.equation_side %x, %y : tuple<!bmodelica.int, !bmodelica.int>
                %rhs = bmodelica.equation_side %x, %y : tuple<!bmodelica.int, !bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int, !bmodelica.int>, tuple<!bmodelica.int, !bmodelica.int>
            }
        }
    }
}

// -----

// COM: An equation whose LHS side carries more than one value must be rejected.

bmodelica.model @MultipleLHSValues {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    bmodelica.dynamic {
        // expected-error @below {{expected exactly one LHS value in each branch equation}}
        bmodelica.if_equation if {
            %cond = bmodelica.constant #bmodelica<bool true> : !bmodelica.bool
            bmodelica.yield %cond : !bmodelica.bool
        } then {
            bmodelica.equation {
                %x   = bmodelica.variable.get @x : !bmodelica.int
                %y   = bmodelica.variable.get @y : !bmodelica.int
                %lhs = bmodelica.equation_side %x, %y : tuple<!bmodelica.int, !bmodelica.int>
                %rhs = bmodelica.equation_side %x, %y : tuple<!bmodelica.int, !bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int, !bmodelica.int>, tuple<!bmodelica.int, !bmodelica.int>
            }
        } else {
            bmodelica.equation {
                %x   = bmodelica.variable.get @x : !bmodelica.int
                %y   = bmodelica.variable.get @y : !bmodelica.int
                %lhs = bmodelica.equation_side %x, %y : tuple<!bmodelica.int, !bmodelica.int>
                %rhs = bmodelica.equation_side %x, %y : tuple<!bmodelica.int, !bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int, !bmodelica.int>, tuple<!bmodelica.int, !bmodelica.int>
            }
        }
    }
}

// -----

// COM: An equation whose LHS is not a direct variable.get must be rejected.
// COM: Variant: the non-variable.get is in the else-branch; the then-branch is
// COM: well-formed.

bmodelica.model @NonVariableGetLHSInElse {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    bmodelica.dynamic {
        // expected-error @below {{all branches of an if_equation must write to the same left-hand side}}
        bmodelica.if_equation if {
            %cond = bmodelica.constant #bmodelica<bool true> : !bmodelica.bool
            bmodelica.yield %cond : !bmodelica.bool
        } then {
            bmodelica.equation {
                %x   = bmodelica.variable.get @x : !bmodelica.int
                %one = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
                %lhs = bmodelica.equation_side %x   : tuple<!bmodelica.int>
                %rhs = bmodelica.equation_side %one : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        } else {
            bmodelica.equation {
                %one = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
                %two = bmodelica.constant #bmodelica<int 2> : !bmodelica.int
                %lhs = bmodelica.equation_side %one : tuple<!bmodelica.int>
                %rhs = bmodelica.equation_side %two : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        }
    }
}

// -----

// COM: An equation whose LHS is not a variable.get or der(variable.get) must be rejected.
// COM: Variant: the non-variable.get is in the then-branch.

bmodelica.model @NonVariableGetLHS {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    bmodelica.dynamic {
        // expected-error @below {{all branches of an if_equation must write to the same left-hand side}}
        bmodelica.if_equation if {
            %cond = bmodelica.constant #bmodelica<bool true> : !bmodelica.bool
            bmodelica.yield %cond : !bmodelica.bool
        } then {
            bmodelica.equation {
                %one = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
                %two = bmodelica.constant #bmodelica<int 2> : !bmodelica.int
                %lhs = bmodelica.equation_side %one : tuple<!bmodelica.int>
                %rhs = bmodelica.equation_side %two : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        } else {
            bmodelica.equation {
                %x    = bmodelica.variable.get @x : !bmodelica.int
                %zero = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
                %lhs  = bmodelica.equation_side %x    : tuple<!bmodelica.int>
                %rhs  = bmodelica.equation_side %zero : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        }
    }
}
