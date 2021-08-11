#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include "Equation.h"

namespace modelica::codegen::model
{
	/**
	 * This class contains a non-trivial block of the BLT matrix. It contains the
	 * equations from an algebraic loop, an implicit equation or an equation with
	 * a derivative operation.
	 */
	class BltBlock
	{
		private:
		template<typename T>
		using Container = llvm::SmallVector<T, 3>;

		public:
		BltBlock(llvm::ArrayRef<Equation> equations);

		[[nodiscard]] Container<Equation>& getEquations();
		[[nodiscard]] const Container<Equation>& getEquations() const;
		void addEquation(Equation equation);

		[[nodiscard]] size_t equationsCount() const;

		[[nodiscard]] bool isForward() const;
		void setForward(bool isForward);

		[[nodiscard]] size_t size() const;

		private:
		Container<Equation> equations;
		bool isForwardDirection;
	};
}	 // namespace modelica::codegen::model
