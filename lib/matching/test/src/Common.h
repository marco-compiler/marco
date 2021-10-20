#ifndef MARCO_MATCHING_TEST_COMMON_H
#define MARCO_MATCHING_TEST_COMMON_H

#include <llvm/ADT/StringRef.h>
#include <marco/matching/Range.h>

namespace marco::matching
{
	class Variable
	{
		public:
		using Id = llvm::StringRef;

		Variable(llvm::StringRef name, llvm::ArrayRef<long> dimensions = llvm::None)
				: name(name.str()), dimensions(dimensions.begin(), dimensions.end())
		{
			if (this->dimensions.empty())
				this->dimensions.emplace_back(1);
		}

		Id getId() const
		{
			return getName();
		}

		unsigned int getRank() const
		{
			return dimensions.size();
		}

		long getDimensionSize(size_t index) const
		{
			return dimensions[index];
		}

		llvm::StringRef getName() const
		{
			return name;
		}

		private:
		std::string name;
		llvm::SmallVector<long, 3> dimensions;
	};

	class Equation
	{
		public:
		using Id = llvm::StringRef;

		Equation(llvm::StringRef name) : name(name.str())
		{
		}

		Id getId() const
		{
			return getName();
		}

		unsigned int getNumOfIterationVars() const
		{
			return ranges.size();
		}

		Range getIterationRange(size_t index) const
		{
			return ranges[index];
		}

		void addIterationRange(Range range)
		{
			ranges.push_back(range);
		}

		void getVariableAccesses(llvm::SmallVectorImpl<Access<Variable>>& v) const
		{
			v.insert(v.begin(), this->accesses.begin(), this->accesses.end());
		}

		void addVariableAccess(Access<Variable> access)
		{
			accesses.push_back(access);
		}

		llvm::StringRef getName() const
		{
			return name;
		}

		private:
		std::string name;
		llvm::SmallVector<Range, 3> ranges;
		llvm::SmallVector<Access<Variable>, 3> accesses;
	};
}

#endif	// MARCO_MATCHING_TEST_COMMON_H
