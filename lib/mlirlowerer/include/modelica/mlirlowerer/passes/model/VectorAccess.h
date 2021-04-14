#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>
#include <modelica/utils/IndexSet.hpp>
#include <modelica/utils/Interval.hpp>

namespace modelica::codegen::model
{
	class Expression;

	class SingleDimensionAccess
	{
		public:
		SingleDimensionAccess();

		bool operator==(const SingleDimensionAccess& other) const;

		void dump() const;
		void dump(llvm::raw_ostream& os) const;

		[[nodiscard]] int64_t getOffset() const;
		[[nodiscard]] size_t getInductionVar() const;

		[[nodiscard]] bool isOffset() const;
		[[nodiscard]] bool isDirecAccess() const;

		[[nodiscard]] Interval map(const MultiDimInterval& multiDimInterval) const;
		[[nodiscard]] Interval map(const Interval& interval) const;
		[[nodiscard]] size_t map(llvm::ArrayRef<size_t> interval) const;

		static SingleDimensionAccess absolute(int64_t absVal);
		static SingleDimensionAccess relative(int64_t relativeVal, size_t indVar);

		/**
		 * Single dimensions access can be built from expression in the form
		 * v[K], v[i] and v[i +/- K], with i induction variable, that is either
		 * constant access, induction access, or sum of induction + constant
		 * access.
		 */
		[[nodiscard]] static bool isCanonical(const Expression& expression);

		private:
		SingleDimensionAccess(int64_t value, bool isAbs, size_t inductionVar = 0);

		int64_t value;
		size_t inductionVar;
		bool isAbs;
	};

	class VectorAccess
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<SingleDimensionAccess>::iterator;
		using const_iterator = Container<SingleDimensionAccess>::const_iterator;

		VectorAccess(Container<SingleDimensionAccess> vector = { SingleDimensionAccess::absolute(0) });

		[[nodiscard]] bool operator==(const VectorAccess& other) const;
		[[nodiscard]] bool operator!=(const VectorAccess& other) const;

		[[nodiscard]] VectorAccess operator*(const VectorAccess& other) const;
		[[nodiscard]] IndexSet operator*(const IndexSet& other) const;

		void dump() const;
		void dump(llvm::raw_ostream& os) const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		[[nodiscard]] const Container<SingleDimensionAccess>& getMappingOffset() const;

		[[nodiscard]] bool isIdentity() const;

		[[nodiscard]] size_t mappableDimensions() const;

		[[nodiscard]] IndexSet map(const IndexSet& indexSet) const;
		[[nodiscard]] MultiDimInterval map(const MultiDimInterval& interval) const;
		[[nodiscard]] llvm::SmallVector<size_t, 3> map(llvm::ArrayRef<size_t> interval) const;

		[[nodiscard]] VectorAccess invert() const;

		[[nodiscard]] VectorAccess combine(const VectorAccess& other) const;
		[[nodiscard]] SingleDimensionAccess combine(const SingleDimensionAccess& other) const;

		/**
		 * A canonical vector access is either a reference, a subscription
		 * or a nested series of subscription operations.
		 */
		[[nodiscard]] static bool isCanonical(Expression expression);

		private:
		Container<SingleDimensionAccess> vectorAccess;
	};

	class AccessToVar
	{
		public:
		AccessToVar(VectorAccess acc, mlir::Value var);

		bool operator==(const AccessToVar& other) const;
		bool operator!=(const AccessToVar& other) const;

		[[nodiscard]] VectorAccess& getAccess();
		[[nodiscard]] const VectorAccess& getAccess() const;

		[[nodiscard]] mlir::Value getVar() const;

		static AccessToVar fromExp(const Expression& expression);

		private:
		VectorAccess access;
		const mlir::Value var;
	};
}	 // namespace modelica
