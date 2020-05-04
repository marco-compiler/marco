#pragma once
#include <limits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
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
		/**
		 * expression must be a at operator, therefore the left hand
		 * is expression rappresenting a vector of some kind, while right must be
		 * either a ind, a sum/subtraction of ind and a constant, or a single scalar
		 */
		static SingleDimensionAccess fromExp(const ModExp& expression);
		SingleDimensionAccess() = default;

		[[nodiscard]] size_t getInductionVar() const { return inductionVar; }
		[[nodiscard]] int64_t getOffset() const { return value; }
		[[nodiscard]] Interval map(const MultiDimInterval& multDim) const
		{
			if (isDirecAccess())
				return Interval(value, value + 1);
			return map(multDim.at(inductionVar));
		}
		[[nodiscard]] Interval map(const Interval& interval) const
		{
			if (isOffset())
				return Interval(interval.min() + value, interval.max() + value);

			return Interval(value, value + 1);
		}
		[[nodiscard]] size_t map(llvm::ArrayRef<size_t> interval) const
		{
			if (isOffset())
				return interval[inductionVar] + value;

			return value;
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
		/**
		 *
		 * single dimensions access can be built from expression in the form
		 * (at V K), (at V (ind K)), and (at (+/- V (ind I) K)) where
		 * V is the vector, K a constant and I the index of the induction variable
		 *
		 * that is either constant access, induction access, or sum of induction +
		 * constant access
		 *
		 */
		[[nodiscard]] static bool isCanonical(const ModExp& expression);

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
				llvm::SmallVector<SingleDimensionAccess, 3>
						vector = { SingleDimensionAccess::absolute(0) })
				: vectorAccess(std::move(vector))
		{
		}

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
		[[nodiscard]] llvm::SmallVector<size_t, 3> map(
				llvm::ArrayRef<size_t> interval) const;

		[[nodiscard]] size_t mappableDimensions() const
		{
			return llvm::count_if(
					vectorAccess, [](const auto& acc) { return acc.isOffset(); });
		}

		[[nodiscard]] VectorAccess invert() const;
		[[nodiscard]] VectorAccess combine(const VectorAccess& other) const;
		[[nodiscard]] VectorAccess operator*(const VectorAccess& other) const
		{
			return combine(other);
		}
		[[nodiscard]] IndexSet operator*(const IndexSet& other) const
		{
			return map(other);
		}
		[[nodiscard]] SingleDimensionAccess combine(
				const SingleDimensionAccess& other) const;
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;
		[[nodiscard]] std::string toString() const;

		[[nodiscard]] bool operator==(const VectorAccess& other) const
		{
			return vectorAccess == other.vectorAccess;
		}
		[[nodiscard]] bool operator!=(const VectorAccess& other) const
		{
			return !(*this == other);
		}
		/**
		 * a canonical vector access is either a reference
		 * or a nested series of (at (at ...) access) operation all of which are
		 * canonical single dimensions access. that is are all in the forms
		 *  (at e (+/- (ind I) K)) | (at e (ind I)) | (at e K)
		 *
		 */
		[[nodiscard]] static bool isCanonical(const ModExp& expression);

		private:
		llvm::SmallVector<SingleDimensionAccess, 3> vectorAccess;
	};

	class AccessToVar
	{
		public:
		AccessToVar(VectorAccess acc, const std::string& ref)
				: access(std::move(acc)), reference(ref)
		{
		}

		static AccessToVar fromExp(const ModExp& expression);

		[[nodiscard]] const VectorAccess& getAccess() const { return access; }
		[[nodiscard]] VectorAccess& getAccess() { return access; }
		[[nodiscard]] const std::string& getVarName() const { return reference; }

		private:
		VectorAccess access;
		const std::string& reference;
	};

}	 // namespace modelica
