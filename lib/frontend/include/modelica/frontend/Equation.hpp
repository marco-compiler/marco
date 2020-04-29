#pragma once

#include "modelica/frontend/Expression.hpp"
namespace modelica
{
	class Equation
	{
		public:
		Equation(Expression leftHand, Expression rightHand)
				: leftHand(std::move(leftHand)), rightHand(std::move(rightHand))
		{
		}
		[[nodiscard]] const Expression& getLeftHand() const { return leftHand; }
		[[nodiscard]] const Expression& getRightHand() const { return rightHand; }

		[[nodiscard]] Expression& getLeftHand() { return leftHand; }
		[[nodiscard]] Expression& getRightHand() { return rightHand; }

		private:
		Expression leftHand;
		Expression rightHand;
	};
}	 // namespace modelica
