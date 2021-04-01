#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/iterator_range.h>
#include <map>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>
#include <modelica/utils/Interval.hpp>
#include <modelica/utils/MapIterator.hpp>

namespace modelica::codegen::model
{
	/**
	 * Given a causalized equation, that is a equation with only a variable on it
	 * left hand, IndexOfEquation is a helper class that extracts the pointer to
	 * the causalized equation as well as the indexes of the variable.
	 */
	class IndexesOfEquation
	{
		public:
		IndexesOfEquation(const Model& model, const Equation& equation)
				: access(equation.getDeterminedVariable()),
					invertedAccess(access.getAccess().invert()),
					indexSet(access.getAccess().map(equation.getInductions())),

					variable(&model.getVariable(access.getVar())),
					equation(&equation)
		{
		}

		[[nodiscard]] const MultiDimInterval& getInterval() const
		{
			return indexSet;
		}
		[[nodiscard]] const Variable& getVariable() const { return *variable; }
		[[nodiscard]] const Equation& getEquation() const { return *equation; }
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
		const Variable* variable;
		const Equation* equation;
	};

	/**
	 * Matched equation lookup is helper class that given a model, that behaves as
	 * a multimap from variables to IndexesOfEquations that are causalizing that
	 * variable.
	 */
	class MatchedEquationLookup
	{
		using Map = std::multimap<const Variable*, IndexesOfEquation>;
		using iterator = MapIterator<Map::iterator, IndexesOfEquation>;
		using const_iterator = MapIterator<Map::const_iterator, const IndexesOfEquation>;
		using iterator_range = llvm::iterator_range<iterator>;
		using const_iterator_range = llvm::iterator_range<const_iterator>;

		public:
		MatchedEquationLookup(const Model& model)
		{
			for (auto& equation : model)
				addEquation(equation, model);
		}

		MatchedEquationLookup(const Model& model, llvm::ArrayRef<Equation> equs)
		{
			for (auto& equation : equs)
				addEquation(equation, model);
		}

		void addEquation(const Equation& equation, const Model& model)
		{
			IndexesOfEquation index(model, equation);
			const Variable* var = &index.getVariable();
			variables.emplace(var, std::move(index));
		}

		[[nodiscard]] const_iterator_range eqsDeterminingVar(const Variable& var) const
		{
			auto range = variables.equal_range(&var);
			return llvm::make_range(range.first, range.second);
		}

		[[nodiscard]] iterator_range eqsDeterminingVar(const Variable& var)
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
