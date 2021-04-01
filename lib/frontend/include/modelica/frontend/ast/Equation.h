#pragma once

#include <llvm/Support/raw_ostream.h>

#include "Expression.h"

namespace modelica::frontend
{
	class Equation
	{
		public:
		Equation(SourcePosition location, Expression leftHand, Expression rightHand);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] Expression& getLeftHand();
		[[nodiscard]] const Expression& getLeftHand() const;
		void setLeftHand(Expression expression);

		[[nodiscard]] Expression& getRightHand();
		[[nodiscard]] const Expression& getRightHand() const;
		void setRightHand(Expression expression);

		private:
		SourcePosition location;
		Expression leftHand;
		Expression rightHand;
	};
}
