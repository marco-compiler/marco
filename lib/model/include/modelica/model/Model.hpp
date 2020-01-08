#pragma once
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModVariable.hpp"

namespace modelica
{
	class Model
	{
		public:
		Model(
				llvm::SmallVector<ModEquation, 3> equations,
				llvm::StringMap<ModVariable> vars)
				: equations(std::move(equations)), vars(std::move(vars))
		{
		}
		Model(
				std::vector<ModEquation> equationsV, llvm::StringMap<ModVariable> vars)
				: vars(std::move(vars))
		{
			for (auto& m : equationsV)
				equations.push_back(std::move(m));
		}
		Model() = default;
		Model& operator=(Model&& other) = delete;
		Model& operator=(const Model& other) = delete;
		Model(Model&& other) = default;
		Model(const Model& other) = default;

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
		}

		void emplaceEquation(
				ModExp left, ModExp right, llvm::SmallVector<InductionVar, 2> vars = {})
		{
			equations.emplace_back(
					std::move(left), std::move(right), std::move(vars));
		}

		bool addVar(ModVariable exp)
		{
			auto name = exp.getName();
			if (vars.find(name) != vars.end())
				return false;

			vars.try_emplace(move(name), std::move(exp));
			return true;
		}

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

		[[nodiscard]] auto& getEquations() { return equations; }

		[[nodiscard]] const auto& getEquations() const { return equations; }

		protected:
		~Model() = default;

		private:
		llvm::SmallVector<ModEquation, 3> equations;
		llvm::StringMap<ModVariable> vars;
	};

}	 // namespace modelica
