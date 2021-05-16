#pragma once

#include <set>
#include <variant>

#include "llvm/ADT/StringMap.h"
#include "modelica/model/ModBltBlock.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModVariable.hpp"

namespace modelica
{
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

		bool addVar(ModVariable exp)
		{
			if (variables.find(exp.getName()) != variables.end())
				return false;
			auto name = exp.getName();
			variables.try_emplace(std::move(name), std::move(exp));
			return true;
		}

		void addUpdate(std::variant<ModEquation, ModBltBlock> update)
		{
			updates.push_back(std::move(update));
			if (std::holds_alternative<ModEquation>(updates.back()))
				addTemplate(std::get<ModEquation>(updates.back()));
		}

		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		[[nodiscard]] auto& getVars() { return variables; }
		[[nodiscard]] const auto& getVars() const { return variables; }
		[[nodiscard]] auto& getUpdates() { return updates; }
		[[nodiscard]] const auto& getUpdates() const { return updates; }

		private:
		void addTemplate(const ModEquation& eq);
		llvm::StringMap<ModVariable> variables;
		llvm::SmallVector<std::variant<ModEquation, ModBltBlock>, 3> updates;
		std::set<std::shared_ptr<ModEqTemplate>> templates;
	};
}	 // namespace modelica
