#pragma once
#include "llvm/ADT/SmallVector.h"
#include "modelica/model/ModExp.hpp"
#include "modelica/utils/IndexSet.hpp"

namespace modelica
{
	class Displacement
	{
		public:
		Displacement(int64_t value, bool isAbs, size_t inductioVar = 0)
				: value(value), inductionVar(inductioVar), isAbs(isAbs)
		{
		}
		Displacement() = default;

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

		private:
		int64_t value;
		size_t inductionVar;
		bool isAbs;
	};

	class VectorAccess
	{
		public:
		template<typename T>
		VectorAccess(const std::string& referredVar, T vector)
				: vectorAccess(std::move(vector)), referredVar(referredVar)
		{
		}
		VectorAccess(const std::string& referredVar): referredVar(referredVar) {}

		[[nodiscard]] const llvm::SmallVector<Displacement, 3>& getMappingOffset()
				const
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

		private:
		llvm::SmallVector<Displacement, 3> vectorAccess;
		const std::string& referredVar;
	};

	llvm::Optional<Displacement> toDisplacement(const ModExp& expression);

	llvm::Optional<VectorAccess> toVectorAccess(const ModExp& expression);

}	 // namespace modelica
