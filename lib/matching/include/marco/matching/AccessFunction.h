#ifndef MARCO_MATCHING_VECTORACCESSFUNCTION_H
#define MARCO_MATCHING_VECTORACCESSFUNCTION_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

namespace marco::matching
{
	class SingleDimensionAccess
	{
		private:
		SingleDimensionAccess(bool constantAccess, int64_t position, unsigned int inductionVariableIndex = 0);

		public:
		static SingleDimensionAccess constant(int64_t position);
		static SingleDimensionAccess relative(int64_t relativePosition, unsigned int inductionVariableIndex);

		bool isConstantAccess() const;

		int64_t getPosition() const;

		int64_t getOffset() const;
		unsigned int getInductionVariableIndex() const;

		private:
		bool constantAccess;
		int64_t position;
		unsigned int inductionVariableIndex;
	};

	class AccessFunction
	{
		public:
		AccessFunction(llvm::ArrayRef<SingleDimensionAccess> functions);

		private:
		llvm::SmallVector<SingleDimensionAccess, 3> functions;
	};
}

#endif	// MARCO_MATCHING_VECTORACCESSFUNCTION_H
