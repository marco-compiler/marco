#ifndef MARCO_MODELING_TEST_MATCHINGCOMMON_H
#define MARCO_MODELING_TEST_MATCHINGCOMMON_H

#include <llvm/ADT/StringRef.h>
#include <marco/modeling/Matching.h>

namespace marco::modeling::matching::test
{
	class Variable
	{
		public:
		using Id = std::string;

		Variable(llvm::StringRef name, llvm::ArrayRef<long> dimensions = llvm::None);

		Id getId() const;

		unsigned int getRank() const;

		long getDimensionSize(size_t index) const;

		llvm::StringRef getName() const;

		private:
		std::string name;
		llvm::SmallVector<long, 3> dimensions;
	};

	class Equation
	{
		public:
		using Id = std::string;

		Equation(llvm::StringRef name);

		Id getId() const;

		unsigned int getNumOfIterationVars() const;

		long getRangeStart(size_t index) const;

    long getRangeEnd(size_t index) const;

		void addIterationRange(internal::Range range);

		void getVariableAccesses(llvm::SmallVectorImpl<Access<Variable>>& v) const;

		void addVariableAccess(Access<Variable> access);

		llvm::StringRef getName() const;

		private:
		std::string name;
		llvm::SmallVector<internal::Range, 3> ranges;
		llvm::SmallVector<Access<Variable>, 3> accesses;
	};
}

#endif	// MARCO_MODELING_TEST_MATCHINGCOMMON_H
