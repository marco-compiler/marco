#pragma once
#include <numeric>
#include <optional>
#include <set>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/iterator_range.h"
#include "marco/model/Assigment.hpp"
#include "modelica/model/ModBltBlock.hpp"
#include "marco/model/ModEqTemplate.hpp"
#include "marco/model/ModEquation.hpp"
#include "marco/model/ModExp.hpp"
#include "marco/model/ModExpPath.hpp"
#include "marco/model/ModVariable.hpp"

namespace marco
{
	/**
	 * This class contains all the information about the modelica model. This
	 * includes: all the variables, all the equations, all the blt block (if any),
	 * and the templates to print the model. This class is used for the constant
	 * folding and matching phases, it will then be used for the scheduling phase.
	 */
	class Model
	{
		public:
		Model(
				llvm::SmallVector<ModEquation, 3> equs,
				llvm::StringMap<ModVariable> vars);
		Model() = default;

		auto begin() { return equations.begin(); }
		auto end() { return equations.end(); };

		[[nodiscard]] auto begin() const { return equations.begin(); }
		[[nodiscard]] auto end() const { return equations.end(); };

		[[nodiscard]] ModEquation& getEquation(size_t index)
		{
			return equations[index];
		}
		[[nodiscard]] const ModEquation& getEquation(size_t index) const
		{
			return equations[index];
		}

		auto varbegin() { return vars.begin(); }
		auto varend() { return vars.end(); };

		[[nodiscard]] auto varbegin() const { return vars.begin(); }
		[[nodiscard]] auto varend() const { return vars.end(); };

		[[nodiscard]] bool containsVar(llvm::StringRef name) const
		{
			return vars.find(name) != vars.end();
		}

		[[nodiscard]] ModVariable& getVar(llvm::StringRef name)
		{
			assert(containsVar(name));	// NOLINT
			return vars.find(name)->second;
		}

		[[nodiscard]] const ModVariable& getVar(llvm::StringRef name) const
		{
			assert(containsVar(name));	// NOLINT
			return vars.find(name)->second;
		}

		void addEquation(ModEquation equation)
		{
			equations.push_back(std::move(equation));
			addTemplate(equations.back());
		}

		void emplaceEquation(
				ModExp left,
				ModExp right,
				std::string templateName,
				MultiDimInterval vars,
				std::optional<EquationPath> path = std::nullopt)
		{
			equations.emplace_back(
					std::move(left),
					std::move(right),
					std::move(templateName),
					std::move(vars),
					true,
					std::move(path));
			addTemplate(equations.back());
		}

		bool addVar(ModVariable exp);

		template<typename... Args>
		bool emplaceVar(std::string name, Args&&... args)
		{
			if (vars.find(name) != vars.end())
				return false;

			vars.try_emplace(name, name, std::forward<Args>(args)...);
			return true;
		}

		[[nodiscard]] const llvm::StringMap<ModVariable>& getVars() const
		{
			return vars;
		}

		[[nodiscard]] llvm::StringMap<ModVariable>& getVars() { return vars; }
		[[nodiscard]] size_t startingIndex(const std::string& varName) const;

		[[nodiscard]] auto& getEquations() { return equations; }

		[[nodiscard]] const auto& getEquations() const { return equations; }
		[[nodiscard]] size_t equationsCount() const
		{
			size_t count = 0;
			for (const auto& eq : equations)
				count += eq.getInductions().size();

			return count;
		}

		[[nodiscard]] size_t stateCount() const
		{
			size_t count = 0;
			for (const auto& var : vars)
				if (var.second.isState())
					count += var.second.toIndexSet().size();
			return count;
		}

		[[nodiscard]] size_t nonStateNonConstCount() const
		{
			size_t count = 0;
			for (const auto& var : vars)
				if (!var.second.isState() && !var.second.isConstant())
					count += var.second.toIndexSet().size();
			return count;
		}

		auto bltBlocksBegin() { return bltBlocks.begin(); }
		auto bltBlocksEnd() { return bltBlocks.end(); };

		[[nodiscard]] auto bltBlocksBegin() const { return bltBlocks.begin(); }
		[[nodiscard]] auto bltBlocksEnd() const { return bltBlocks.end(); };

		[[nodiscard]] ModBltBlock& getBltBlock(size_t index)
		{
			return bltBlocks[index];
		}
		[[nodiscard]] const ModBltBlock& getBltBlock(size_t index) const
		{
			return bltBlocks[index];
		}

		void addBltBlock(ModBltBlock bltBlock)
		{
			bltBlocks.push_back(std::move(bltBlock));
			addTemplate(bltBlocks.back());
		}

		[[nodiscard]] auto& getBltBlocks() { return bltBlocks; }
		[[nodiscard]] const auto& getBltBlocks() const { return bltBlocks; }

		using TemplateEqMap = std::set<std::shared_ptr<ModEqTemplate>>;
		using TemplateBltMap = std::set<std::shared_ptr<ModBltTemplate>>;

		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		private:
		void addTemplate(const ModEquation& eq);
		void addTemplate(const ModBltBlock& bltBlock);
		llvm::SmallVector<ModEquation, 3> equations;
		llvm::StringMap<ModVariable> vars;
		llvm::SmallVector<ModBltBlock, 3> bltBlocks;
		TemplateEqMap eqTemplates;
		TemplateBltMap bltTemplates;
	};

}	 // namespace marco
