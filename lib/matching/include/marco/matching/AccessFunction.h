#ifndef MARCO_MATCHING_VECTORACCESSFUNCTION_H
#define MARCO_MATCHING_VECTORACCESSFUNCTION_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

namespace marco::matching
{
	class SingleDimensionAccess
	{
		private:
		SingleDimensionAccess(bool constantAccess, long position, unsigned int inductionVariableIndex = 0);

		public:
		static SingleDimensionAccess constant(long position);
		static SingleDimensionAccess relative(unsigned int inductionVariableIndex, long relativePosition);

		size_t operator()(llvm::ArrayRef<long> equationIndexes) const;

		bool isConstantAccess() const;

		size_t getPosition() const;

		size_t getOffset() const;
		unsigned int getInductionVariableIndex() const;

		private:
		bool constantAccess;
		long position;
		unsigned int inductionVariableIndex;
	};

	class AccessFunction
	{
		private:
		using Container = llvm::SmallVector<SingleDimensionAccess, 3>;

		public:
		using iterator = Container::iterator;
		using const_iterator = Container::const_iterator;

		AccessFunction(llvm::ArrayRef<SingleDimensionAccess> functions);

		SingleDimensionAccess operator[](size_t index) const;

		llvm::ArrayRef<SingleDimensionAccess> getDimensionAccesses() const;

		void map(llvm::SmallVectorImpl<size_t>& results, llvm::ArrayRef<long> equationIndexes) const;

		size_t size() const;

		iterator begin();
		const_iterator begin() const;

		iterator end();
		const_iterator end() const;

		private:
		Container functions;
	};
}

#endif	// MARCO_MATCHING_VECTORACCESSFUNCTION_H
