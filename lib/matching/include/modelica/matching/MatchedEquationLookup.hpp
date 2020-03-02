#pragma once
#include <map>

#include "llvm/ADT/iterator_range.h"
#include "modelica/model/Model.hpp"
#include "modelica/utils/MapIterator.hpp"

namespace modelica
{
	class IndexesOfEquation
	{
		public:
		IndexesOfEquation(const Model& model, const ModEquation& equation)
				: equation(&equation)
		{
			auto access = equation.getDeterminedVariable();
			indexSet = access.getAccess().map(equation.toIndexSet());
			variable = &model.getVar(access.getVarName());
		}

		[[nodiscard]] const IndexSet& getIndexSet() const { return indexSet; }
		[[nodiscard]] const ModVariable& getVariable() const { return *variable; }
		[[nodiscard]] const ModEquation& getEquation() const { return *equation; }

		private:
		IndexSet indexSet;
		const ModVariable* variable;
		const ModEquation* equation;
	};

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
