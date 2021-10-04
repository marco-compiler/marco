#pragma once

#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Expression.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/Variable.h>
#include <marco/mlirlowerer/passes/model/VectorAccess.h>
#include <mlir/Support/LogicalResult.h>
#include <sundials/sundials_types.h>

namespace marco::codegen::ida
{
	/**
	 * This class is used for testing and debugging purposes only. Given a
	 * modelica model, it uses the SUNDIALS IDA library to solve non-trivial
	 * blocks of the BLT matrix.
	 */
	class IdaSolver
	{
		using Dimension = std::vector<std::pair<size_t, size_t>>;

		public:
		IdaSolver(
				const model::Model &model,
				realtype startTime = 0.0,
				realtype stopTime = 10.0,
				realtype relativeTolerance = 1e-6,
				realtype absoluteTolerance = 1e-6);

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

		[[nodiscard]] sunindextype getForEquationsNumber();
		[[nodiscard]] sunindextype getEquationsNumber();
		[[nodiscard]] sunindextype getNonZeroValuesNumber();
		[[nodiscard]] realtype getTime();
		[[nodiscard]] realtype getVariable(sunindextype index);
		[[nodiscard]] realtype getDerivative(sunindextype index);
		[[nodiscard]] sunindextype getRowLength(sunindextype index);
		[[nodiscard]] Dimension getDimension(sunindextype index);

		private:
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
		sunindextype getFunction(const model::Expression &exp);

		private:
		// Model data
		const model::Model model;
		const realtype stopTime;
		sunindextype forEquationsNumber;

		std::map<model::Variable, realtype> initialValueMap;
		std::map<model::Variable, sunindextype> variableIndexMap;
		std::map<std::pair<model::Variable, model::VectorAccess>, sunindextype>
				accessesMap;
		void *userData;
	};
}	 // namespace marco::codegen::ida
