#pragma once
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <map>
#include <variant>

#include "llvm/ADT/iterator_range.h"
#include "marco/model/ModBltBlock.hpp"
#include "marco/model/ModEquation.hpp"
#include "marco/model/Model.hpp"
#include "marco/model/VectorAccess.hpp"
#include "marco/utils/Interval.hpp"
#include "marco/utils/MapIterator.hpp"

namespace marco
{
	/**
	 * This class can be built with an equation or a BLT block.
	 * Given a causalized equation, that is a equation with only a variable on it
	 * left hand, IndexOfEquation is a helper class that extracts the pointer to
	 * the causalized equation as well as the indexes of the variable.
	 * Given a BLT block, IndexOfEquation extracts the pointers to all the
	 * causalized equations inside the BLT block, along with the indexes of the
	 * variable.
	 */
	class IndexesOfEquation
	{
		public:
		IndexesOfEquation(const Model& model, const ModEquation& equation);

		IndexesOfEquation(const Model& model, const ModBltBlock& bltBlock);

		[[nodiscard]] bool isEquation() const
		{
			return std::holds_alternative<ModEquation>(content);
		}
		[[nodiscard]] bool isBltBlock() const
		{
			return std::holds_alternative<ModBltBlock>(content);
		}

		[[nodiscard]] const std::variant<ModEquation, ModBltBlock>& getContent()
				const
		{
			return content;
		}

		[[nodiscard]] const ModEquation& getEquation() const
		{
			assert(isEquation());
			return std::get<ModEquation>(content);
		}
		[[nodiscard]] const ModBltBlock& getBltBlock() const
		{
			assert(isBltBlock());
			return std::get<ModBltBlock>(content);
		}

		[[nodiscard]] const auto& getEquations() const { return equations; }
		[[nodiscard]] const auto& getVariables() const { return variables; }
		[[nodiscard]] const auto& getEqToVars() const { return directAccesses; }
		[[nodiscard]] const auto& getVarToEqs() const { return invertedAccesses; }
		[[nodiscard]] const auto& getIntervals() const { return indexSets; }

		[[nodiscard]] const ModVariable* getVariable() const
		{
			assert(isEquation());
			return variables.front();
		}
		[[nodiscard]] const VectorAccess& getEqToVar() const
		{
			assert(isEquation());
			return directAccesses.front();
		}
		[[nodiscard]] const VectorAccess& getVarToEq() const
		{
			assert(isEquation());
			return invertedAccesses.front();
		}
		[[nodiscard]] const MultiDimInterval& getInterval() const
		{
			assert(isEquation());
			return indexSets.front();
		}

		[[nodiscard]] size_t size() const { return equations.size(); }

		private:
		const std::variant<ModEquation, ModBltBlock> content;
		const llvm::SmallVector<ModEquation, 3> equations;
		llvm::SmallVector<AccessToVar, 3> accesses;
		llvm::SmallVector<const ModVariable*, 3> variables;
		llvm::SmallVector<VectorAccess, 3> directAccesses;
		llvm::SmallVector<VectorAccess, 3> invertedAccesses;
		llvm::SmallVector<MultiDimInterval, 3> indexSets;
	};

	/**
	 * Matched equation lookup is helper class that given a model, that behaves as
	 * a multimap from variables to IndexesOfEquations that are causalizing that
	 * variable.
	 */
	class MatchedEquationLookup
	{
		using Map = std::multimap<const ModVariable*, IndexesOfEquation*>;
		using iterator = MapIterator<Map::iterator, IndexesOfEquation*>;
		using const_iterator =
				MapIterator<Map::const_iterator, const IndexesOfEquation*>;
		using iterator_range = llvm::iterator_range<iterator>;
		using const_iterator_range = llvm::iterator_range<const_iterator>;

		public:
		MatchedEquationLookup(const Model& model);

		MatchedEquationLookup(const Model& model, llvm::ArrayRef<ModEquation> equs);

		void addEquation(const ModEquation& equation, const Model& model);

		void addBltBlock(const ModBltBlock& bltBlock, const Model& model);

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
		[[nodiscard]] size_t size() const { return variables.size(); }

		private:
		Map variables;
	};
}	 // namespace marco
