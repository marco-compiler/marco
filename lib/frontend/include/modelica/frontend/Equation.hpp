#pragma once

#include <llvm/Support/raw_ostream.h>

#include "Expression.hpp"

namespace modelica
{
	class Equation
	{
		public:
		Equation(Expression leftHand, Expression rightHand);

		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] Expression& getLeftHand();
		[[nodiscard]] Expression& getRightHand();

		[[nodiscard]] const Expression& getLeftHand() const;
		[[nodiscard]] const Expression& getRightHand() const;

		private:
		Expression leftHand;
		Expression rightHand;
	};
}	 // namespace modelica
