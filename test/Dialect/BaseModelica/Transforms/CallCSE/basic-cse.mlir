// RUN: modelica-opt %s --call-cse -split-input-file | FileCheck %s

// CHECK-LABEL: @SimpleExpression
module @SimpleExpression {
	bmodelica.function @foo {
		bmodelica.variable @x : !bmodelica.variable<f64, input>
		bmodelica.variable @y : !bmodelica.variable<f64, output>

		bmodelica.algorithm {
			%0 = bmodelica.variable_get @x : f64
			bmodelica.variable_set @y, %0 : f64
		}
	}

    // CHECK: bmodelica.model
	bmodelica.model @M {
		// CHECK: bmodelica.variable @[[CSE:_cse0_0]]
		bmodelica.variable @x : !bmodelica.variable<f64>
		bmodelica.variable @y : !bmodelica.variable<f64>

		%t0 = bmodelica.equation_template inductions = [] {
			%0 = bmodelica.variable_get @x : f64
			%lhs = bmodelica.equation_side %0 : tuple<f64>
			%1 = bmodelica.constant 1.0 : f64
			// CHECK: %[[RES0:.*]] = bmodelica.variable_get @[[CSE]]
			// CHECK-NEXT: %[[LHS0:.*]] = bmodelica.equation_side %[[RES0]]
			// CHECK-NEXT: bmodelica.equation_sides %{{.*}}, %[[LHS0]]
			%2 = bmodelica.call @foo(%1) : (f64) -> f64
			%rhs = bmodelica.equation_side %2 : tuple<f64>
			bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
		}

		%t1 = bmodelica.equation_template inductions = [] {
			%0 = bmodelica.variable_get @y : f64
			%lhs = bmodelica.equation_side %0 : tuple<f64>
			%1 = bmodelica.constant 1.0 : f64
			// CHECK: %[[RES1:.*]] = bmodelica.variable_get @[[CSE]]
			// CHECK-NEXT: %[[LHS1:.*]] = bmodelica.equation_side %[[RES1]]
			// CHECK-NEXT: bmodelica.equation_sides %{{.*}}, %[[LHS1]]
			%2 = bmodelica.call @foo(%1) : (f64) -> f64
			%rhs = bmodelica.equation_side %2 : tuple<f64>
			bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
		}

		bmodelica.dynamic {
			bmodelica.equation_instance %t0 : !bmodelica.equation
			bmodelica.equation_instance %t1 : !bmodelica.equation
		}

		// CHECK: %[[TEMPLATE:.*]] = bmodelica.equation_template
		// CHECK-NEXT: %[[RES2:.*]] = bmodelica.variable_get @[[CSE]]
		// CHECK-NEXT: %[[LHS2:.*]] = bmodelica.equation_side %[[RES2]]
		// CHECK-NEXT: %[[RES3:.*]] = bmodelica.constant 1
		// CHECK-NEXT: %[[RES4:.*]] = bmodelica.call @foo(%[[RES3]])
		// CHECK-NEXT: %[[RHS:.*]] = bmodelica.equation_side %[[RES4]]
		// CHECK-NEXT: bmodelica.equation_sides %[[LHS2]], %[[RHS]]

		// CHECK: bmodelica.dynamic
		// CHECK-NEXT: bmodelica.equation_instance %[[TEMPLATE]]
	}
}

// -----

// CHECK-LABEL: @ComplexExpression
module @ComplexExpression {
	bmodelica.function @foo {
		bmodelica.variable @x : !bmodelica.variable<f64, input>
		bmodelica.variable @y : !bmodelica.variable<f64, output>

		bmodelica.algorithm {
			%0 = bmodelica.variable_get @x : f64
			bmodelica.variable_set @y, %0 : f64
		}
	}

    // CHECK: bmodelica.model
	bmodelica.model @M {
		// CHECK: bmodelica.variable @[[CSE:_cse0_0]]
		bmodelica.variable @x : !bmodelica.variable<f64>
		bmodelica.variable @y : !bmodelica.variable<f64>

		%t0 = bmodelica.equation_template inductions = [] {
			%0 = bmodelica.variable_get @x : f64
			%lhs = bmodelica.equation_side %0 : tuple<f64>
			%c1 = bmodelica.constant 1.0 : f64
			%c2 = bmodelica.constant 2.0 : f64
			%c3 = bmodelica.constant 3.0 : f64
			%sub = bmodelica.sub %c2, %c1 : (f64, f64) -> f64
			%cos = bmodelica.cos %sub : f64 -> f64
			%pow = bmodelica.pow %cos, %c3 : (f64, f64) -> f64
			// CHECK: %[[RES0:.*]] = bmodelica.variable_get @[[CSE]]
			// CHECK-NEXT: %[[LHS0:.*]] = bmodelica.equation_side %[[RES0]]
			// CHECK-NEXT: bmodelica.equation_sides %{{.*}}, %[[LHS0]]
			%2 = bmodelica.call @foo(%pow) : (f64) -> f64
			%rhs = bmodelica.equation_side %2 : tuple<f64>
			bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
		}

		%t1 = bmodelica.equation_template inductions = [] {
			%0 = bmodelica.variable_get @y : f64
			%lhs = bmodelica.equation_side %0 : tuple<f64>
			%c1 = bmodelica.constant 1.0 : f64
			%c2 = bmodelica.constant 2.0 : f64
			%c3 = bmodelica.constant 3.0 : f64
			%sub = bmodelica.sub %c2, %c1 : (f64, f64) -> f64
			%cos = bmodelica.cos %sub : f64 -> f64
			%pow = bmodelica.pow %cos, %c3 : (f64, f64) -> f64
			// CHECK: %[[RES1:.*]] = bmodelica.variable_get @[[CSE]]
			// CHECK-NEXT: %[[LHS1:.*]] = bmodelica.equation_side %[[RES1]]
			// CHECK-NEXT: bmodelica.equation_sides %{{.*}}, %[[LHS1]]
			%2 = bmodelica.call @foo(%pow) : (f64) -> f64
			%rhs = bmodelica.equation_side %2 : tuple<f64>
			bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
		}

		bmodelica.dynamic {
			bmodelica.equation_instance %t0 : !bmodelica.equation
			bmodelica.equation_instance %t1 : !bmodelica.equation
		}

		// CHECK: %[[TEMPLATE:.*]] = bmodelica.equation_template
		// CHECK-NEXT: %[[RES2:.*]] = bmodelica.variable_get @[[CSE]]
		// CHECK-NEXT: %[[LHS2:.*]] = bmodelica.equation_side %[[RES2]]
		// CHECK-DAG: %[[C1:.*]] = bmodelica.constant 1
		// CHECK-DAG: %[[C2:.*]] = bmodelica.constant 2
		// CHECK-DAG: %[[C3:.*]] = bmodelica.constant 3
		// CHECK-DAG: %[[SUB:.*]] = bmodelica.sub %[[C2]], %[[C1]]
		// CHECK-DAG: %[[COS:.*]] = bmodelica.cos %[[SUB]]
		// CHECK-DAG: %[[POW:.*]] = bmodelica.pow %[[COS]], %[[C3]]
		// CHECK-NEXT: %[[RES4:.*]] = bmodelica.call @foo(%[[POW]])
		// CHECK-NEXT: %[[RHS:.*]] = bmodelica.equation_side %[[RES4]]
		// CHECK-NEXT: bmodelica.equation_sides %[[LHS2]], %[[RHS]]

		// CHECK: bmodelica.dynamic
		// CHECK-NEXT: bmodelica.equation_instance %[[TEMPLATE]]
	}
}
