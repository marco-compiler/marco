#pragma once
#include <limits>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/model/ModExp.hpp"
#include "modelica/utils/IndexSet.hpp"

namespace modelica
{
	class SingleDimensionAccess
	{
		public:
		static SingleDimensionAccess absolute(int64_t absVal)
		{
			return SingleDimensionAccess(absVal, true);
		}
		static SingleDimensionAccess relative(int64_t relativeVal, size_t indVar)
		{
			return SingleDimensionAccess(relativeVal, false, indVar);
		}
		SingleDimensionAccess() = default;

		[[nodiscard]] size_t getInductionVar() const { return inductionVar; }
		[[nodiscard]] int64_t getOffset() const { return value; }
		[[nodiscard]] Interval map(const MultiDimInterval& multDim) const
		{
			return map(multDim.at(inductionVar));
		}
		[[nodiscard]] Interval map(const Interval& interval) const
		{
			if (isOffset())
				return Interval(interval.min() + value, interval.max() + value);

			return Interval(value, value + 1);
		}

		[[nodiscard]] bool isOffset() const { return !isAbs; }
		[[nodiscard]] bool isDirecAccess() const { return isAbs; }
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		bool operator==(const SingleDimensionAccess& other) const
		{
			if (isAbs != other.isAbs)
				return false;
			if (isAbs)
				return value == other.value;
			return value == other.value && inductionVar == other.inductionVar;
		}

		private:
		SingleDimensionAccess(int64_t value, bool isAbs, size_t inductioVar = 0)
				: value(value), inductionVar(inductioVar), isAbs(isAbs)
		{
		}
		int64_t value{ 0 };
		size_t inductionVar{ std::numeric_limits<size_t>::max() };
		bool isAbs{ true };
	};

	class VectorAccess
	{
		public:
		VectorAccess(
				const std::string& referredVar,
				llvm::SmallVector<SingleDimensionAccess, 3> vector)
				: vectorAccess(std::move(vector)), referredVar(referredVar)
		{
		}
		VectorAccess(const std::string& referredVar): referredVar(referredVar) {}

		[[nodiscard]] const llvm::SmallVector<SingleDimensionAccess, 3>&
		getMappingOffset() const
		{
			return vectorAccess;
		}

		[[nodiscard]] IndexSet map(const IndexSet& indexSet) const
		{
			IndexSet toReturn;

			for (const auto& part : indexSet)
				toReturn.unite(map(part));

			return toReturn;
		}

		[[nodiscard]] MultiDimInterval map(const MultiDimInterval& interval) const;
		[[nodiscard]] const std::string& getName() const { return referredVar; }

		[[nodiscard]] size_t mappableDimensions() const
		{
			size_t count = 0;
			for (const auto& acc : vectorAccess)
				if (acc.isOffset())
					count++;

			return count;
		}

		[[nodiscard]] VectorAccess invert() const;
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;
		[[nodiscard]] std::string toString() const;

		[[nodiscard]] bool operator==(const VectorAccess& other) const
		{
			return vectorAccess == other.vectorAccess &&
						 referredVar == other.referredVar;
		}
		[[nodiscard]] bool operator!=(const VectorAccess& other) const
		{
			return !(*this == other);
		}

		private:
		llvm::SmallVector<SingleDimensionAccess, 3> vectorAccess;
		const std::string& referredVar;
	};

	bool isCanonicalSingleDimensionAccess(const ModExp& expresion);
	bool isCanonicalVectorAccess(const ModExp& expression);
	SingleDimensionAccess toSingleDimensionAccess(const ModExp& expression);

	VectorAccess toVectorAccess(const ModExp& expression);

}	 // namespace modelica
