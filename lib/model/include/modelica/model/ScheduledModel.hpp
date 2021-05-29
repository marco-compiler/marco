#pragma once

#include <set>
#include <variant>

#include "llvm/ADT/StringMap.h"
#include "modelica/model/ModBltBlock.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModVariable.hpp"

namespace modelica
{
	/**
	 * This class contains all the information about the modelica model. This
	 * includes: all the variables, all the equations and all the blt block (in
	 * the order in which they need to be executed), and the templates to print
	 * the model. This class is obtained from the scheduling phase and will be
	 * passed to the solver.
	 */
	class ScheduledModel
	{
		public:
		ScheduledModel(llvm::StringMap<ModVariable> variables);

		ScheduledModel() = default;

		[[nodiscard]] bool containsVar(llvm::StringRef name) const
		{
			return variables.find(name) != variables.end();
		}

		[[nodiscard]] ModVariable& getVar(llvm::StringRef name)
		{
			assert(containsVar(name));	// NOLINT
			return variables.find(name)->second;
		}

		[[nodiscard]] const ModVariable& getVar(llvm::StringRef name) const
		{
			assert(containsVar(name));	// NOLINT
			return variables.find(name)->second;
		}

		void addUpdate(std::variant<ModEquation, ModBltBlock> update)
		{
			updates.push_back(std::move(update));
			addTemplate(updates.back());
		}

		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		[[nodiscard]] auto& getVars() { return variables; }
		[[nodiscard]] const auto& getVars() const { return variables; }
		[[nodiscard]] auto& getUpdates() { return updates; }
		[[nodiscard]] const auto& getUpdates() const { return updates; }

		using TemplateMap = std::set<std::variant<
				std::shared_ptr<ModEqTemplate>,
				std::shared_ptr<ModBltTemplate>>>;

		private:
		void addTemplate(const std::variant<ModEquation, ModBltBlock>& update);
		llvm::StringMap<ModVariable> variables;
		llvm::SmallVector<std::variant<ModEquation, ModBltBlock>, 3> updates;
		TemplateMap templates;
	};
}	 // namespace modelica
