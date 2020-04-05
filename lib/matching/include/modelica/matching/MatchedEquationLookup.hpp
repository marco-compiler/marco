#pragma once
#include <map>

#include "llvm/ADT/iterator_range.h"
#include "modelica/model/Model.hpp"
#include "modelica/model/VectorAccess.hpp"
#include "modelica/utils/Interval.hpp"
#include "modelica/utils/MapIterator.hpp"

namespace modelica
{
	/**
	 * Given a causalized equation, that is a equation with only a variable on it
	 * left hand, IndexOfEquation is a helper class that extracts the pointer to
	 * the causalized equation as well as the indexes of the variable.
	 */
	class IndexesOfEquation
	{
		public:
		IndexesOfEquation(const Model& model, const ModEquation& equation)
				: access(equation.getDeterminedVariable()),
					invertedAccess(access.getAccess().invert()),
					indexSet(access.getAccess().map(equation.getInductions())),

					variable(&model.getVar(access.getVarName())),
					equation(&equation)
		{
		}

		[[nodiscard]] const MultiDimInterval& getInterval() const
		{
			return indexSet;
		}
		[[nodiscard]] const ModVariable& getVariable() const { return *variable; }
		[[nodiscard]] const ModEquation& getEquation() const { return *equation; }
		[[nodiscard]] const VectorAccess& getEqToVar() const
		{
			return access.getAccess();
		}
		[[nodiscard]] const VectorAccess& getVarToEq() const
		{
			return invertedAccess;
		}

		private:
		AccessToVar access;
		VectorAccess invertedAccess;
		MultiDimInterval indexSet;
		const ModVariable* variable;
		const ModEquation* equation;
	};

	/**
	 * Matched equation lookup is helper class that given a model, that behaves as
	 * a multimap from variables to IndexesOfEquations that are causalizing that
	 * variable.
	 */
	class MatchedEquationLookup
	{
		using Map = std::multimap<const ModVariable*, IndexesOfEquation>;
		using iterator = MapIterator<Map::iterator, IndexesOfEquation>;
		using const_iterator =
				MapIterator<Map::const_iterator, const IndexesOfEquation>;
		using iterator_range = llvm::iterator_range<iterator>;
		using const_iterator_range = llvm::iterator_range<const_iterator>;

		public:
		MatchedEquationLookup(const Model& model)
		{
			for (auto& equation : model)
			{
				IndexesOfEquation index(model, equation);
				const ModVariable* var = &index.getVariable();
				variables.emplace(var, std::move(index));
			}
		}

		[[nodiscard]] const_iterator_range eqsDeterminingVar(
				const ModVariable& var) const
		{
			auto range = variables.equal_range(&var);
			return llvm::make_range(range.first, range.second);
		}

		[[nodiscard]] iterator_range eqsDeterminingVar(const ModVariable& var)
		{
			auto range = variables.equal_range(&var);
			return llvm::make_range(range.first, range.second);
		}

		[[nodiscard]] iterator begin() { return variables.begin(); }
		[[nodiscard]] iterator end() { return variables.end(); }
		[[nodiscard]] const_iterator begin() const { return variables.begin(); }
		[[nodiscard]] const_iterator end() const { return variables.end(); }

		private:
		Map variables;
	};
}	 // namespace modelica
