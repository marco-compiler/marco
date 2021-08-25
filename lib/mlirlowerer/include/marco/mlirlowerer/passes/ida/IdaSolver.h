#pragma once

#include <mlir/Support/LogicalResult.h>
#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Expression.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/Variable.h>

namespace marco::codegen::ida
{
	class IdaSolver
	{
		public:
		IdaSolver(
				model::Model &model,
				double startTime = 0.0,
				double stopTime = 10.0,
				double relativeTolerance = 1e-6,
				double absoluteTolerance = 1e-6);

		/**
		 * Instantiate and initialize all the classes and data needed by IDA to
		 * solve the given system of equations. It must be called before the first
		 * usage of step(). It may fail in case of malformed model.
		 */
		[[nodiscard]] mlir::LogicalResult init();

		/**
		 * Invokes IDA to perform one step of the computation. Returns 1 if the
		 * computation has not reached the 'stopTime' seconds limit, 0 if it has
		 * reached the end of the computation, -1 if it fails.
		 */
		[[nodiscard]] int8_t step();

		/**
		 * Performs a full run of the system.
		 */
		[[nodiscard]] mlir::LogicalResult run(llvm::raw_ostream &OS = llvm::outs());

		/**
		 * Frees all the data allocated by IDA.
		 */
		[[nodiscard]] mlir::LogicalResult free();

		/**
		 * Prints the current time of the computation and the value all variables to
		 * the given stream.
		 */
		void printOutput(llvm::raw_ostream &OS = llvm::outs());

		/**
		 * Prints statistics about the computation of the system.
		 */
		void printStats(llvm::raw_ostream &OS = llvm::outs());

		[[nodiscard]] size_t getProblemSize();
		[[nodiscard]] int64_t getEquationsNumber();
		[[nodiscard]] double getTime();
		[[nodiscard]] double getVariable(size_t index);
		[[nodiscard]] double getDerivative(size_t index);
		[[nodiscard]] size_t getRowLength(size_t index);
		[[nodiscard]] std::vector<std::pair<size_t, size_t>> getDimension(
				size_t index);

		private:
		[[nodiscard]] int64_t computeNEQ();
		[[nodiscard]] int64_t computeNNZ();

		/**
		 * Given an equation from the model, returns the multidimensional interval
		 * through which the equation iterates. Every interval is 0-indexed, with
		 * the first extreme included and last extreme excluded.
		 */
		void getDimension(const model::Equation &eq);

		/**
		 * Given an equation from the model, creates a lambda function that computes
		 * the residual function and the Jacobian function of that equation starting
		 * from the data provided by IDA, the value of all induction variables and
		 * the variable with respect to which the equation must be differentiated.
		 */
		void getResidualAndJacobian(const model::Equation &eq);

		/**
		 * Given an expression from the model, returns a lambda function that
		 * computes the expression starting from the data provided
		 * by IDA and the value of all induction variables.
		 */
		size_t getFunction(const model::Expression &exp);

		private:
		// Model data
		model::Model model;
		size_t problemSize;
		const int64_t equationsNumber;
		std::map<model::Variable, double> initialValueMap;
		std::map<model::Variable, size_t> indexOffsetMap;
		void *userData;
	};
}	 // namespace marco::codegen::ida
