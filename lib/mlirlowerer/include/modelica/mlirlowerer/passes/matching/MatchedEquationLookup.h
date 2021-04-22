#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/iterator_range.h>
#include <map>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>
#include <modelica/utils/Interval.hpp>
#include <modelica/utils/MapIterator.hpp>

namespace modelica::codegen::model
{
	/**
	 * Given a causalized equation (that is, an equation with only a variable
	 * on its left-hand side), IndexOfEquation is an helper class that extracts
	 * the pointer to the causalized equation as well as the indexes of the
	 * variable.
	 */
	class IndexesOfEquation
	{
		public:
		IndexesOfEquation(const Model& model, Equation equation);

		[[nodiscard]] const Equation& getEquation() const;
		[[nodiscard]] const Variable& getVariable() const;
		[[nodiscard]] const VectorAccess& getEqToVar() const;
		[[nodiscard]] const VectorAccess& getVarToEq() const;
		[[nodiscard]] const MultiDimInterval& getInterval() const;

		private:
		Equation equation;
		AccessToVar access;
		VectorAccess invertedAccess;
		MultiDimInterval indexSet;
		Variable variable;
	};

	/**
	 * Matched equation lookup is helper class that given a model, that
	 * behaves as a multimap from variables to IndexesOfEquations that are
	 * causalizing that variable.
	 */
	class MatchedEquationLookup
	{
		using Map = std::multimap<Variable, IndexesOfEquation>;
		using iterator = MapIterator<Map::iterator, IndexesOfEquation>;
		using const_iterator = MapIterator<Map::const_iterator, const IndexesOfEquation>;
		using iterator_range = llvm::iterator_range<iterator>;
		using const_iterator_range = llvm::iterator_range<const_iterator>;

		public:
		MatchedEquationLookup(const Model& model);
		MatchedEquationLookup(const Model& model, llvm::ArrayRef<Equation> equations);

		void addEquation(Equation equation, const Model& model);

		[[nodiscard]] iterator_range eqsDeterminingVar(const Variable& var);
		[[nodiscard]] const_iterator_range eqsDeterminingVar(const Variable& var) const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		Map variables;
	};
}
