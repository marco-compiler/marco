#pragma once

#include "llvm/Support/raw_ostream.h"
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

		void dump(llvm::raw_ostream& OS, size_t indents = 0) const
		{
			OS << "equation\n";
			leftHand.dump(OS, indents + 1);
			rightHand.dump(OS, indents + 1);
		}

		private:
		Expression leftHand;
		Expression rightHand;
	};
}	 // namespace modelica
