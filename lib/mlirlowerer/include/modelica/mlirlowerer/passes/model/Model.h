#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <mlir/IR/Value.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <set>

namespace modelica::codegen::model
{
	class Equation;
	class EquationTemplate;
	class Variable;

	class Model
	{
		private:
		template<typename T> using Container = llvm::SmallVector<std::shared_ptr<T>, 3>;
		using TemplateMap = std::set<std::shared_ptr<EquationTemplate>>;

		public:
		using iterator = boost::indirect_iterator<Container<Equation>::iterator>;
		using const_iterator = boost::indirect_iterator<Container<Equation>::const_iterator>;

		Model(SimulationOp op,
					llvm::ArrayRef<std::shared_ptr<Variable>> variables,
					llvm::ArrayRef<std::shared_ptr<Equation>> equations);

		static Model build(SimulationOp op);

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		[[nodiscard]] SimulationOp getOp() const;

		[[nodiscard]] bool hasVariable(mlir::Value var) const;
		Variable& getVariable(mlir::Value var);
		const Variable& getVariable(mlir::Value var) const;
		Container<Variable>& getVariables();
		const Container<Variable>& getVariables() const;
		void addVariable(mlir::Value var);

		[[nodiscard]] Container<Equation>& getEquations();
		[[nodiscard]] const Container<Equation>& getEquations() const;
		void addEquation(Equation equation);

		[[nodiscard]] const TemplateMap& getTemplates() const;

		/**
		 * Get the number of the equations that will compose the final model.
		 */
		[[nodiscard]] size_t equationsCount() const;

		/**
		 * Get the amount of variables that are not state or constant ones.
		 */
		[[nodiscard]] size_t nonStateNonConstCount() const;

		private:
		SimulationOp op;
		Container<Variable> variables;
		Container<Equation> equations;
		TemplateMap templates;
	};
}