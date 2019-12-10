#pragma once
#include "modelica/model/Assigment.hpp"

namespace modelica
{
	class AssignModel
	{
		public:
		AssignModel(
				llvm::StringMap<ModExp> vars,
				llvm::SmallVector<Assigment, 0> updates = {})
				: variables(std::move(vars)), updates(std::move(updates))
		{
		}

		AssignModel() = default;

		/**
		 * adds a var to the simulation that will be intialized with the provided
		 * expression. Notice that in the initialization is undefined behaviour to
		 * use references to other variables.
		 *
		 * \return true if there were no other vars with the same name already.
		 */
		[[nodiscard]] bool addVar(std::string name, ModExp exp)
		{
			if (variables.find(name) != variables.end())
				return false;
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
		}

		template<typename... T>
		void emplaceUpdate(T&&... args)
		{
			updates.emplace_back(std::forward<T>(args)...);
		}
		void dump(llvm::raw_ostream& OS = llvm::outs()) const
		{
			OS << "vars\n";
			for (const auto& var : variables)
			{
				OS << var.first() << " = ";
				var.second.dump(OS);

				OS << "\n";
			}

			OS << "equations\n";
			for (const auto& update : updates)
				update.dump(OS);
		}

		[[nodiscard]] const auto& getVars() const { return variables; }
		[[nodiscard]] auto& getVars() { return variables; }
		[[nodiscard]] auto& getUpdates() { return updates; }
		[[nodiscard]] const auto& getUpdates() const { return updates; }

		private:
		llvm::StringMap<ModExp> variables;
		llvm::SmallVector<Assigment, 0> updates;
	};
}	 // namespace modelica
