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

	bmodelica.model @M {
		bmodelica.variable @x : !bmodelica.variable<f64>
		bmodelica.variable @y : !bmodelica.variable<f64>
		bmodelica.variable @z : !bmodelica.variable<f64>

		//%t0 = bmodelica.equation_template inductions = [] {
		//	%0 = bmodelica.variable_get @x : f64
		//	%lhs = bmodelica.equation_side %0 : tuple<f64>
		//	%1 = bmodelica.constant 23.0 : f64
		//	%2 = bmodelica.call @foo(%1) : (f64) -> f64
		//	%rhs = bmodelica.equation_side %2 : tuple<f64>
		//	bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
		//}

		//%t1 = bmodelica.equation_template inductions = [] {
		//	%0 = bmodelica.variable_get @y : f64
		//	%lhs = bmodelica.equation_side %0 : tuple<f64>
		//	%1 = bmodelica.constant 23.0 : f64
		//	%2 = bmodelica.call @foo(%1) : (f64) -> f64
		//	%rhs = bmodelica.equation_side %2 : tuple<f64>
		//	bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
		//}

		%t0 = bmodelica.equation_template inductions = [] {
			%0 = bmodelica.variable_get @x : f64
			%lhs = bmodelica.equation_side %0 : tuple<f64>
			%1 = bmodelica.constant 23.0 : f64
			%2 = bmodelica.constant 25.0 : f64
			%3 = bmodelica.add %1, %2 : (f64, f64) -> f64
			%4 = bmodelica.call @foo(%3) : (f64) -> f64
			%rhs = bmodelica.equation_side %4 : tuple<f64>
			bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
		}

		%t1 = bmodelica.equation_template inductions = [] {
			%0 = bmodelica.variable_get @y : f64
			%lhs = bmodelica.equation_side %0 : tuple<f64>
			%1 = bmodelica.constant 23.0 : f64
			%2 = bmodelica.constant 25.0 : f64
			%3 = bmodelica.add %1, %2 : (f64, f64) -> f64
			%4 = bmodelica.call @foo(%3) : (f64) -> f64
			%rhs = bmodelica.equation_side %4 : tuple<f64>
			bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
		}

		%t2 = bmodelica.equation_template inductions = [] {
			%0 = bmodelica.variable_get @z : f64
			%lhs = bmodelica.equation_side %0 : tuple<f64>
			%1 = bmodelica.constant 57.0 : f64
			%2 = bmodelica.call @foo(%1) : (f64) -> f64
			%rhs = bmodelica.equation_side %2 : tuple<f64>
			bmodelica.equation_sides %lhs, %rhs : tuple<f64>, tuple<f64>
		}

		bmodelica.dynamic {
			bmodelica.equation_instance %t0 : !bmodelica.equation
			bmodelica.equation_instance %t1 : !bmodelica.equation
			bmodelica.equation_instance %t2 : !bmodelica.equation
		}
	}
}
