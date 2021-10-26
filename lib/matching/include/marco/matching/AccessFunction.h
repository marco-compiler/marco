#ifndef MARCO_MATCHING_VECTORACCESSFUNCTION_H
#define MARCO_MATCHING_VECTORACCESSFUNCTION_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

namespace marco::matching
{
	class DimensionAccess
	{
		private:
		DimensionAccess(bool constantAccess, long position, unsigned int inductionVariableIndex = 0);

		public:
		static DimensionAccess constant(long position);
		static DimensionAccess relative(unsigned int inductionVariableIndex, long relativePosition);

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
		using Container = llvm::SmallVector<DimensionAccess, 3>;

		public:
		using iterator = Container::iterator;
		using const_iterator = Container::const_iterator;

		AccessFunction(llvm::ArrayRef<DimensionAccess> functions);

		DimensionAccess operator[](size_t index) const;

		llvm::ArrayRef<DimensionAccess> getDimensionAccesses() const;

		void map(llvm::SmallVectorImpl<long>& results, llvm::ArrayRef<long> equationIndexes) const;

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
