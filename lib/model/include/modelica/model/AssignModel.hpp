#pragma once
#include <set>
#include "llvm/ADT/StringMap.h"

#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModVariable.hpp"

namespace modelica
{
	class AssignModel
	{
		public:
		AssignModel(
				llvm::StringMap<ModVariable> vars,
				llvm::SmallVector<Assigment, 2> ups = {})
				: variables(std::move(vars)), updates(std::move(ups))
		{
			for (const auto& update : updates)
				addTemplate(update);
		}

		AssignModel() = default;

		/**
		 * adds a var to the simulation that will be intialized with the provided
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
		 * Add an update expression for a particular variable.
		 * notice that if a expression is referring to a missing
		 * variable then it's lower that will fail, not addUpdate
		 *
		 */
		void addUpdate(Assigment assigment)
		{
			updates.push_back(std::move(assigment));
			addTemplate(updates.back());
		}

		template<typename... T>
		void emplaceUpdate(T&&... args)
		{
			updates.emplace_back(std::forward<T>(args)...);
			addTemplate(updates.back());
		}
		void dump(llvm::raw_ostream& OS = llvm::outs()) const
		{
			OS << "init\n";
			for (const auto& var : variables)
			{
				OS << var.first() << " = ";
				var.second.getInit().dump(OS);

				OS << "\n";
			}

			if (!templates.empty())
				OS << "templates\n";
			for (const auto& pair : templates)
			{
				pair->dump(true, OS);
				OS << "\n";
			}

			OS << "update\n";
			for (const auto& update : updates)
				update.dump(OS);
		}

		[[nodiscard]] const auto& getVars() const { return variables; }
		[[nodiscard]] auto& getVars() { return variables; }
		[[nodiscard]] auto& getUpdates() { return updates; }
		[[nodiscard]] const auto& getUpdates() const { return updates; }

		private:
		void addTemplate(const Assigment& assigment)
		{
			if (assigment.getTemplate()->getName().empty())
				return;

			if (templates.find(assigment.getTemplate()) != templates.end())
				return;

			templates.emplace(assigment.getTemplate());
		}

		llvm::StringMap<ModVariable> variables;
		llvm::SmallVector<Assigment, 2> updates;
		std::set<std::shared_ptr<ModEqTemplate>> templates;
	};
}	 // namespace modelica
