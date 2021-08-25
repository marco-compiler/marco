#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/iterator_range.h>
#include <map>
#include <marco/mlirlowerer/passes/model/BltBlock.h>
#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/Variable.h>
#include <marco/mlirlowerer/passes/model/VectorAccess.h>
#include <marco/utils/Interval.hpp>
#include <marco/utils/MapIterator.hpp>
#include <variant>

namespace marco::codegen::model
{
	/**
	 * This class can be built with an equation or a BLT block.
	 * Given a causalized equation (that is, an equation with only a variable
	 * on its left-hand side), IndexOfEquation is an helper class that extracts
	 * the pointer to the causalized equation as well as the indexes of the
	 * variable.
	 * Given a BLT block, IndexOfEquation extracts the pointers to all the
	 * causalized equations inside the BLT block, along with the indexes of the
	 * variable.
	 */
	class IndexesOfEquation
	{
		public:
		IndexesOfEquation(const Model& model, Equation equation);
		IndexesOfEquation(const Model& model, BltBlock equation);

		[[nodiscard]] bool isEquation() const;
		[[nodiscard]] bool isBltBlock() const;

		[[nodiscard]] const std::variant<Equation, BltBlock>& getContent() const;
		[[nodiscard]] const Equation& getEquation() const;
		[[nodiscard]] const BltBlock& getBltBlock() const;

		[[nodiscard]] const Variable& getVariable() const;
		[[nodiscard]] const VectorAccess& getEqToVar() const;
		[[nodiscard]] const VectorAccess& getVarToEq() const;
		[[nodiscard]] const MultiDimInterval& getInterval() const;

		[[nodiscard]] const llvm::SmallVector<Equation, 3>& getEquations() const;
		[[nodiscard]] const llvm::SmallVector<Variable, 3>& getVariables() const;
		[[nodiscard]] const llvm::SmallVector<VectorAccess, 3>& getEqToVars() const;
		[[nodiscard]] const llvm::SmallVector<VectorAccess, 3>& getVarToEqs() const;
		[[nodiscard]] const llvm::SmallVector<MultiDimInterval, 3>& getIntervals() const;

		[[nodiscard]] size_t size() const;

		private:
		const std::variant<Equation, BltBlock> content;
		const llvm::SmallVector<Equation, 3> equations;
		llvm::SmallVector<AccessToVar, 3> accesses;
		llvm::SmallVector<Variable, 3> variables;
		llvm::SmallVector<VectorAccess, 3> directAccesses;
		llvm::SmallVector<VectorAccess, 3> invertedAccesses;
		llvm::SmallVector<MultiDimInterval, 3> indexSets;
	};

	/**
	 * Matched equation lookup is helper class that given a model, that
	 * behaves as a multimap from variables to IndexesOfEquations that are
	 * causalizing that variable.
	 */
	class MatchedEquationLookup
	{
		using Map = std::multimap<Variable, IndexesOfEquation*>;
		using iterator = MapIterator<Map::iterator, IndexesOfEquation*>;
		using const_iterator = MapIterator<Map::const_iterator, const IndexesOfEquation*>;
		using iterator_range = llvm::iterator_range<iterator>;
		using const_iterator_range = llvm::iterator_range<const_iterator>;

		public:
		MatchedEquationLookup(const Model& model);
		MatchedEquationLookup(const Model& model, llvm::ArrayRef<Equation> equations);

		void addEquation(Equation equation, const Model& model);
		void addBltBlock(BltBlock bltBlock, const Model& model);

		[[nodiscard]] iterator_range eqsDeterminingVar(const Variable& var);
		[[nodiscard]] const_iterator_range eqsDeterminingVar(const Variable& var) const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		Map variables;
	};
}	 // namespace marco::codegen::model
