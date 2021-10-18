#ifndef MARCO_MATCHING_REFERENCEMATCHER_H
#define MARCO_MATCHING_REFERENCEMATCHER_H

#include <llvm/ADT/SmallVector.h>

namespace marco::matching
{
	template<
	    class Equation,
			class Expression>
	class ReferenceMatcher
	{
		public:
		ReferenceMatcher(Equation equation)
		{
			visit(equation.getLhs());
			visit(equation.getRhs());
		}

		void visit(Expression expression)
		{

		}

		private:
		llvm::SmallVector<size_t, 3> currentPath;
		llvm::SmallVector<ExpressionPath, 3> pathsToVariables;
	};
}

#endif	// MARCO_MATCHING_REFERENCEMATCHER_H
