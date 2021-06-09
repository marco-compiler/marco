#pragma once

#include <memory>
#include <set>

#include "modelica/model/ModBltTemplate.hpp"
#include "modelica/model/ModEqTemplate.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExp.hpp"

namespace modelica
{
	/**
	 * This class contains an Algebraic Loop not solvable by the SccCollapsing
	 * pass. This must be solved with the Substitution method or with a solver
	 * like IDA.
	 */
	class ModBltBlock
	{
		public:
		ModBltBlock(llvm::SmallVector<ModEquation, 3> equs, std::string bltName);
		ModBltBlock() = default;

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

		[[nodiscard]] auto& getEquations() { return equations; }
		[[nodiscard]] const auto& getEquations() const { return equations; }

		[[nodiscard]] size_t equationsCount() const
		{
			size_t count = 0;
			for (const ModEquation& eq : equations)
				count += eq.getInductions().size();

			return count;
		}

		using TemplateMap = std::set<std::shared_ptr<ModEqTemplate>>;
		using ResidualFunction = llvm::SmallVector<ModExp, 3>;
		using JacobianMatrix = llvm::SmallVector<llvm::SmallVector<ModExp, 3>, 3>;

		[[nodiscard]] ResidualFunction& getResidual() { return residualFunction; }
		[[nodiscard]] const ResidualFunction& getResidual() const
		{
			return residualFunction;
		}

		[[nodiscard]] JacobianMatrix& getJacobian() { return jacobianMatrix; }
		[[nodiscard]] const JacobianMatrix& getJacobian() const
		{
			return jacobianMatrix;
		}

		[[nodiscard]] bool isForward() const { return isForwardDirection; }
		void setForward(bool isForward) { isForwardDirection = isForward; }

		[[nodiscard]] auto& getTemplate() { return body; }
		[[nodiscard]] const auto& getTemplate() const { return body; }
		[[nodiscard]] const std::string& getTemplateName() const
		{
			return body->getName();
		}

		[[nodiscard]] size_t size() const;

		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		private:
		void addTemplate(const ModEquation& eq);
		void computeResidualFunction();
		void computeJacobianMatrix();

		llvm::SmallVector<ModEquation, 3> equations;
		TemplateMap templates;
		ResidualFunction residualFunction;
		JacobianMatrix jacobianMatrix;
		bool isForwardDirection = true;
		std::shared_ptr<ModBltTemplate> body;
	};
}	 // namespace modelica
