#pragma once

#include <set>

#include "llvm/ADT/StringMap.h"
#include "modelica/model/ModEqTemplate.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModVariable.hpp"

namespace modelica
{
	/**
	 * This class contains an Algebraic Loop not solvable by the SccCollapsing
	 * pass. This must be solved with the Substitution method or with a solver
	 * like IDA.
	 */
	class ModLoop
	{
		public:
		ModLoop(
				llvm::SmallVector<ModEquation, 3> equs,
				llvm::StringMap<ModVariable> vars);
		ModLoop() = default;

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

		[[nodiscard]] const llvm::StringMap<ModVariable>& getVars() const
		{
			return vars;
		}

		[[nodiscard]] llvm::StringMap<ModVariable>& getVars() { return vars; }

		[[nodiscard]] auto& getEquations() { return equations; }
		[[nodiscard]] const auto& getEquations() const { return equations; }

		[[nodiscard]] size_t equationsCount() const
		{
			size_t count = 0;
			for (const auto& eq : equations)
				count += eq.getInductions().size();

			return count;
		}

		using TemplateMap = std::set<std::shared_ptr<ModEqTemplate>>;

		[[nodiscard]] TemplateMap& getTemplates() { return templates; }

		[[nodiscard]] const TemplateMap& getTemplates() const { return templates; }

		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		private:
		void addTemplate(const ModEquation& eq);
		llvm::SmallVector<ModEquation, 3> equations;
		llvm::StringMap<ModVariable> vars;
		TemplateMap templates;
	};
}	 // namespace modelica
