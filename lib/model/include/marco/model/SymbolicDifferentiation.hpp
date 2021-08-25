#pragma once

#include "marco/model/ModEquation.hpp"
#include "marco/model/ModExp.hpp"

namespace marco
{
	/**
	 * Differentiate the expression with respect to a given variable access.
	 *
	 * @param expression Expression to be differentiated.
	 * @param variableAccess The variable accces, either scalar or vector, with
	 * respect to which the derivative is computed.
	 *
	 * @return Derivative of the given expression.
	 */
	[[nodiscard]] ModExp differentiate(
			const ModExp& expression, const ModExp& variableAccess);

	/**
	 * Differentiate both left and right hand expressions of an equation with
	 * respect to a given variable access.
	 *
	 * @param equation Equation to be differentiatied.
	 * @param variableAccess The variable accces, either scalar or vector, with
	 * respect to which the derivative is computed.
	 *
	 * @return Derivative of the given equation.
	 */
	[[nodiscard]] ModEquation differentiate(
			const ModEquation& equation, const ModExp& variableAccess);
}	 // namespace marco
