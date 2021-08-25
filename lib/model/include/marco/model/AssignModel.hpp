#pragma once

#include <set>
#include <variant>

#include "llvm/ADT/StringMap.h"
#include "marco/model/Assigment.hpp"
#include "marco/model/ModBltBlock.hpp"
#include "marco/model/ModVariable.hpp"

namespace marco
{
	/**
	 * This class contains all the information about the modelica model. This
	 * includes: all the variables, all the assignments and all the blt block (in
	 * the order in which they need to be executed), and the templates to print
	 * the model. This class is obtained from the solver phase and will be passed
	 * to the lowerer.
	 */
	class AssignModel
	{
		public:
		AssignModel(
				llvm::StringMap<ModVariable> vars,
				llvm::SmallVector<std::variant<Assigment, ModBltBlock>, 2> ups = {})
				: variables(std::move(vars)), updates(std::move(ups))
		{
			for (const auto& update : updates)
				addTemplate(update);
		}

		AssignModel() = default;

		/**
		 * Adds a var to the simulation that will be intialized with the provided
		 * expression. Notice that in the initialization is undefined behaviour to
		 * use references to other variables.
		 *
		 * \return true if there were no other vars with the same name already.
		 */
		[[nodiscard]] bool addVar(ModVariable exp)
		{
			if (variables.find(exp.getName()) != variables.end())
				return false;
			auto name = exp.getName();
			variables.try_emplace(std::move(name), std::move(exp));
			return true;
		}

		/**
		 * Add an update expression for a particular variable. If the parmeter is an
		 * Assignment, it will be lowered as an assignment operation. If it is a
		 * ModBltBlock it will be lowered with the usage of a solver. Notice that if
		 * an expression is referring to a missing variable then its lowering will
		 * fail, not addUpdate.
		 */
		void addUpdate(std::variant<Assigment, ModBltBlock> update)
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
		void addTemplate(const std::variant<Assigment, ModBltBlock>& update);
		llvm::StringMap<ModVariable> variables;
		llvm::SmallVector<std::variant<Assigment, ModBltBlock>, 3> updates;
		TemplateMap templates;
	};
}	 // namespace marco
