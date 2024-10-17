// RUN: modelica-opt %s --call-cse | FileCheck %s

// CHECK-LABEL: @Test
module @Test {
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
		// CHECK-NEXT: bmodelica.variable @[[CSE1:_cse1_0]]
		// CHECK-NEXT: bmodelica.variable @[[CSE0:_cse0_0]]
		bmodelica.variable @x : !bmodelica.variable<f64>
		bmodelica.variable @y : !bmodelica.variable<f64>

		%t0 = bmodelica.equation_template inductions = [] {
			%0 = bmodelica.variable_get @x : f64
			%lhs = bmodelica.equation_side %0 : tuple<f64>
			%c1 = bmodelica.constant 1.0 : f64
			%c2 = bmodelica.constant 2.0 : f64
			%c3 = bmodelica.constant 3.0 : f64
			%add = bmodelica.add %c1, %c2 : (f64, f64) -> f64
			%sub = bmodelica.sub %c2, %c3 : (f64, f64) -> f64
			// CHECK: %[[c0:.*]] = bmodelica.variable_get @[[CSE0]]
			// CHECK-NEXT: %[[c1:.*]] = bmodelica.variable_get @[[CSE1]]
			// CHECK-NEXT: %[[RES:.*]] = bmodelica.add %[[c0]], %[[c1]]
			// CHECK-NEXT: %[[LHS:.*]] = bmodelica.equation_side %[[RES]]
			// CHECK-NEXT: bmodelica.equation_sides %{{.*}}, %[[LHS]]
			%2 = bmodelica.call @foo(%add) : (f64) -> f64
			%3 = bmodelica.call @foo(%sub) : (f64) -> f64
			%4 = bmodelica.add %2, %3 : (f64, f64) -> f64
			%rhs = bmodelica.equation_side %4 : tuple<f64>
			bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
		}

		%t1 = bmodelica.equation_template inductions = [] {
			%0 = bmodelica.variable_get @x : f64
			%lhs = bmodelica.equation_side %0 : tuple<f64>
			%c1 = bmodelica.constant 1.0 : f64
			%c2 = bmodelica.constant 2.0 : f64
			%add = bmodelica.add %c1, %c2 : (f64, f64) -> f64
			// CHECK: %[[c:.*]] = bmodelica.variable_get @[[CSE0]]
			// CHECK-NEXT: %[[LHS:.*]] = bmodelica.equation_side %[[c]]
			// CHECK-NEXT: bmodelica.equation_sides %{{.*}}, %[[LHS]]
			%2 = bmodelica.call @foo(%add) : (f64) -> f64
			%rhs = bmodelica.equation_side %2 : tuple<f64>
			bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
		}

		%t2 = bmodelica.equation_template inductions = [] {
			%0 = bmodelica.variable_get @x : f64
			%lhs = bmodelica.equation_side %0 : tuple<f64>
			%c2 = bmodelica.constant 2.0 : f64
			%c3 = bmodelica.constant 3.0 : f64
			%sub = bmodelica.sub %c2, %c3 : (f64, f64) -> f64
			// CHECK: %[[c:.*]] = bmodelica.variable_get @[[CSE1]]
			// CHECK-NEXT: %[[LHS:.*]] = bmodelica.equation_side %[[c]]
			// CHECK-NEXT: bmodelica.equation_sides %{{.*}}, %[[LHS]]
			%3 = bmodelica.call @foo(%sub) : (f64) -> f64
			%rhs = bmodelica.equation_side %3 : tuple<f64>
			bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
		}

		bmodelica.dynamic {
			bmodelica.equation_instance %t0 : !bmodelica.equation
			bmodelica.equation_instance %t1 : !bmodelica.equation
			bmodelica.equation_instance %t2 : !bmodelica.equation
		}

		// CHECK: %[[T0:.*]] = bmodelica.equation_template
		// CHECK-NEXT: %[[RES0:.*]] = bmodelica.variable_get @[[CSE0]]
		// CHECK-NEXT: %[[LHS:.*]] = bmodelica.equation_side %[[RES0]]
		// CHECK-DAG: %[[RES1:.*]] = bmodelica.constant 1
		// CHECK-DAG: %[[RES2:.*]] = bmodelica.constant 2
		// CHECK-DAG: %[[RES3:.*]] = bmodelica.add %[[RES1]], %[[RES2]]
		// CHECK-NEXT: %[[RES4:.*]] = bmodelica.call @foo(%[[RES3]])
		// CHECK-NEXT: %[[RHS:.*]] = bmodelica.equation_side %[[RES4]]
		// CHECK-NEXT: bmodelica.equation_sides %[[LHS]], %[[RHS]]

		// CHECK: %[[T1:.*]] = bmodelica.equation_template
		// CHECK-NEXT: %[[RES0:.*]] = bmodelica.variable_get @[[CSE1]]
		// CHECK-NEXT: %[[LHS:.*]] = bmodelica.equation_side %[[RES0]]
		// CHECK-DAG: %[[RES1:.*]] = bmodelica.constant 2
		// CHECK-DAG: %[[RES2:.*]] = bmodelica.constant 3
		// CHECK-DAG: %[[RES3:.*]] = bmodelica.sub %[[RES1]], %[[RES2]]
		// CHECK-NEXT: %[[RES4:.*]] = bmodelica.call @foo(%[[RES3]])
		// CHECK-NEXT: %[[RHS:.*]] = bmodelica.equation_side %[[RES4]]
		// CHECK-NEXT: bmodelica.equation_sides %[[LHS]], %[[RHS]]

		// CHECK: bmodelica.dynamic
		// CHECK-DAG: bmodelica.equation_instance %[[T0]]
		// CHECK-DAG: bmodelica.equation_instance %[[T1]]
	}
}
