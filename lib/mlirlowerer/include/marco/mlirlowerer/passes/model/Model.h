#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <mlir/IR/Value.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaDialect.h>
#include <set>

#include "Equation.h"
#include "BltBlock.h"

namespace marco::codegen::model
{
	class Variable;

	class Model
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<Equation>::iterator;
		using const_iterator = Container<Equation>::const_iterator;

		Model() = default;
		Model(modelica::SimulationOp op,
					llvm::ArrayRef<std::shared_ptr<Variable>> variables,
					llvm::ArrayRef<Equation> equations,
					llvm::ArrayRef<BltBlock> bltBlocks = {});

		static Model build(modelica::SimulationOp op);
		void reloadIR();

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		[[nodiscard]] modelica::SimulationOp getOp() const;

		[[nodiscard]] bool hasVariable(mlir::Value var) const;
		[[nodiscard]] Variable getVariable(mlir::Value var) const;
		Container<std::shared_ptr<Variable>>& getVariables();
		const Container<std::shared_ptr<Variable>>& getVariables() const;
		void addVariable(mlir::Value var);

		[[nodiscard]] Container<Equation>& getEquations();
		[[nodiscard]] const Container<Equation>& getEquations() const;
		void addEquation(Equation equation);

		[[nodiscard]] Container<BltBlock>& getBltBlocks();
		[[nodiscard]] const Container<BltBlock>& getBltBlocks() const;
		void addBltBlock(BltBlock bltBlock);

		/**
		 * Get the number of the equations that will compose the final model.
		 */
		[[nodiscard]] size_t equationsCount() const;

		/**
		 * Get the amount of variables that are not state or constant ones.
		 */
		[[nodiscard]] size_t nonStateNonConstCount() const;

		private:
		modelica::SimulationOp op;
		Container<std::shared_ptr<Variable>> variables;
		Container<Equation> equations;
		Container<BltBlock> bltBlocks;
	};
}
