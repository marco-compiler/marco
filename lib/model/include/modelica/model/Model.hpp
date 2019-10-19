#pragma once
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModVariable.hpp"

namespace modelica
{
	class Model
	{
		public:
		Model(std::vector<ModEquation> equations, llvm::StringMap<ModVariable> vars)
				: equations(std::move(equations)), vars(std::move(vars))
		{
		}
		Model() = default;
		Model& operator=(Model&& other) = delete;
		Model& operator=(const Model& other) = delete;
		Model(Model&& other) = default;
		Model(const Model& other) = default;

		auto begin() { return equations.begin(); }
		auto end() { return equations.end(); };

		[[nodiscard]] auto begin() const { return equations.cbegin(); }
		[[nodiscard]] auto end() const { return equations.cend(); };

		[[nodiscard]] ModEquation& getEquation(size_t index)
		{
			return equations.at(index);
		}
		[[nodiscard]] const ModEquation& getEquation(size_t index) const
		{
			return equations.at(index);
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
		}

		template<typename... Args>
		void addEquation(Args... args)
		{
			equations.emplace_back(std::forward<Args>(args)...);
		}

		[[nodiscard]] bool addVar(ModVariable exp)
		{
			auto name = exp.getName();
			if (vars.find(name) != vars.end())
				return false;

			vars.try_emplace(move(name), std::move(exp));
			return true;
		}

		template<typename... Args>
		[[nodiscard]] bool emplaceVar(std::string name, Args... args)
		{
			if (vars.find(name) != vars.end())
				return false;

			vars.try_emplace(name, name, std::forward<Args>(args)...);
			return true;
		}

		protected:
		~Model() = default;

		private:
		std::vector<ModEquation> equations;
		llvm::StringMap<ModVariable> vars;
	};

}	 // namespace modelica
