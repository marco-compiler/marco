#pragma once
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <map>
#include <variant>

#include "llvm/ADT/iterator_range.h"
#include "modelica/model/ModBltBlock.hpp"
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
		IndexesOfEquation(const Model& model, const ModEquation& equation)
				: content(equation),
					accesses({ equation.getDeterminedVariable() }),
					variables({ &model.getVar(accesses.front().getVarName()) }),
					directAccesses({ accesses.front().getAccess() }),
					invertedAccesses({ accesses.front().getAccess().invert() }),
					indexSets(
							{ accesses.front().getAccess().map(equation.getInductions()) })
		{
		}

		IndexesOfEquation(const Model& model, const ModBltBlock& bltBlock)
				: content(bltBlock)
		{
			for (const ModEquation& eq : bltBlock.getEquations())
			{
				accesses.push_back(eq.getDeterminedVariable());
				variables.push_back(&model.getVar(accesses.back().getVarName()));
				directAccesses.push_back(accesses.back().getAccess());
				invertedAccesses.push_back(accesses.back().getAccess().invert());
				indexSets.push_back(directAccesses.back().map(eq.getInductions()));
			}
		}

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

		[[nodiscard]] const auto& getVariables() const
		{
			assert(isBltBlock());
			return variables;
		}
		[[nodiscard]] const auto& getEqToVars() const
		{
			assert(isBltBlock());
			return directAccesses;
		}
		[[nodiscard]] const auto& getVarToEqs() const
		{
			assert(isBltBlock());
			return invertedAccesses;
		}
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

		[[nodiscard]] size_t size() const { return accesses.size(); }

		private:
		const std::variant<ModEquation, ModBltBlock> content;
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
		MatchedEquationLookup(const Model& model)
		{
			for (const ModEquation& equation : model)
				addEquation(equation, model);
			for (const ModBltBlock& bltBlock : model.getBltBlocks())
				addBltBlock(bltBlock, model);
		}

		MatchedEquationLookup(const Model& model, llvm::ArrayRef<ModEquation> equs)
		{
			assert(model.getBltBlocks().empty());
			for (const ModEquation& equation : equs)
				addEquation(equation, model);
		}

		void addEquation(const ModEquation& equation, const Model& model)
		{
			IndexesOfEquation* index = new IndexesOfEquation(model, equation);
			const ModVariable* var = index->getVariable();
			variables.emplace(var, index);
		}

		void addBltBlock(const ModBltBlock& bltBlock, const Model& model)
		{
			IndexesOfEquation* index = new IndexesOfEquation(model, bltBlock);
			for (const ModVariable* var : index->getVariables())
				variables.emplace(var, index);
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
		[[nodiscard]] size_t size() const { return variables.size(); }

		private:
		Map variables;
	};
}	 // namespace marco
