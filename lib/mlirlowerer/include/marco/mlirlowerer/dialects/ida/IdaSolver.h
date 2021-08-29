#pragma once

#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Expression.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/Variable.h>
#include <mlir/Support/LogicalResult.h>

namespace marco::codegen::ida
{
	/**
	 * This class is used for testing and debugging purposes only. Given a
	 * modelica model, it uses the SUNDIALS IDA library to solve non-trivial
	 * blocks of the BLT matrix.
	 */
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
		 * Invoke IDA to perform one step of the computation. Returns false if the
		 * computation failed, true otherwise.
		 */
		[[nodiscard]] bool step();

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

		[[nodiscard]] int64_t getProblemSize();
		[[nodiscard]] int64_t getEquationsNumber();
		[[nodiscard]] double getTime();
		[[nodiscard]] double getVariable(int64_t index);
		[[nodiscard]] double getDerivative(int64_t index);
		[[nodiscard]] int64_t getRowLength(int64_t index);
		[[nodiscard]] std::vector<std::pair<int64_t, int64_t>> getDimension(
				int64_t index);

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
		int64_t getFunction(const model::Expression &exp);

		private:
		// Model data
		model::Model model;
		double stopTime;
		int64_t problemSize;
		const int64_t equationsNumber;
		std::map<model::Variable, double> initialValueMap;
		std::map<model::Variable, int64_t> indexOffsetMap;
		void *userData;
	};
}	 // namespace marco::codegen::ida
