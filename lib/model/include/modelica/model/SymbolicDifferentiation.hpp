#pragma once

#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModVariable.hpp"

namespace modelica
{
	/**
	 * Differentiate the expression with respect to a given variable.
	 *
	 * @param expression Expression to be differentiated.
	 * @param variable The variable with respect to which the derivative is
	 * computed.
	 * @param index The access to the vector variable with respect to which the
	 * derivative is computed.
	 * @return Derivative of the given expression.
	 */
	ModExp differentiate(
			const ModExp& expression,
			const ModVariable& variable,
			const ModExp& index = ModConst());

	/**
	 * Differentiate both left and right hand expressions of an equation with
	 * respect to a given variable.
	 *
	 * @param equation Equation to be differentiatied.
	 * @param variable The variable with respect to which the derivative is
	 * computed.
	 * @param index The access to the vector variable with respect to which the
	 * derivative is computed.
	 * @return Derivative of the given equation.
	 */
	ModEquation differentiate(
			const ModEquation& equation,
			const ModVariable& variable,
			const ModExp& index = ModConst());
}	 // namespace modelica
