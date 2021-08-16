#pragma once

#include <ida/ida.h>
#include <mlir/Support/LogicalResult.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_klu.h>
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunnonlinsol/sunnonlinsol_newton.h>

namespace modelica::codegen::ida
{
	using Dimensions = std::vector<std::pair<size_t, size_t>>;
	using Indexes = std::vector<size_t>;
	using Function = std::function<double(
			double tt, double cj, double *yy, double *yp, Indexes &ind, size_t var)>;

	/**
	 * Container for all the data and lambda functions required by IDA in order to
	 * compute the residual functions and the jacobian matrix.
	 */
	typedef struct UserData
	{
		std::vector<size_t> rowLength;
		std::vector<Dimensions> dimensions;
		std::vector<Function> residuals;
		std::vector<Function> jacobianMatrix;
	} UserData;

	class IdaSolver
	{
		public:
		IdaSolver(
				model::Model &model,
				const realtype startTime = 0.0,
				const realtype stopTime = 10.0,
				const realtype relativeTolerance = 1e-6,
				const realtype absoluteTolerance = 1e-6);

		/**
		 * Instantiate and initialize all the classes and data needed by IDA to
		 * solve the given system of equations. It must be called before the first
		 * usage of step(). It may fail in case of malformed model.
		 */
		[[nodiscard]] mlir::LogicalResult init();

		/**
		 * Invokes IDA to perform one step of the computation. Returns true if the
		 * computation has not reached the 'stopTime' seconds limit, false if it has
		 * reached the end of the computation, std::nullopt if errors occurred.
		 */
		[[nodiscard]] std::optional<bool> step();

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
		mlir::LogicalResult printStats(llvm::raw_ostream &OS = llvm::outs());

		[[nodiscard]] UserData *getData() { return data; }
		[[nodiscard]] realtype getTime() { return time; }
		[[nodiscard]] realtype *getVariables() { return variablesValues; }
		[[nodiscard]] realtype *getDerivatives() { return derivativesValues; }

		private:
		[[nodiscard]] sunindextype computeNEQ();
		[[nodiscard]] sunindextype computeNNZ();

		/**
		 * Initializes the three arrays used by IDA and the UserData struct.
		 */
		void initData();

		/**
		 * Given an equation from the model, returns the multidimensional interval
		 * through which the equation iterates. Every interval is 0-indexed, with
		 * the first extreme included and last extreme excluded.
		 */
		[[nodiscard]] Dimensions getDimensions(const model::Equation &eq);

		/**
		 * Given an equation from the model, returns a lambda function that computes
		 * the residual function of that equation starting from the data provided
		 * by IDA and the value of all induction variables.
		 */
		[[nodiscard]] Function getResidual(const model::Equation &eq);

		/**
		 * Given an equation from the model, returns a lambda function that computes
		 * the Jacobian function of that equation starting from the data provided
		 * by IDA, the value of all induction variables and the variable with
		 * respect to which the equation must be differentiated.
		 */
		[[nodiscard]] Function getJacobian(const model::Equation &eq);

		/**
		 * Given an expression from the model, returns a lambda function that
		 * computes the expression starting from the data provided
		 * by IDA and the value of all induction variables.
		 */
		[[nodiscard]] Function getFunction(const model::Expression &exp);

		/**
		 * Given an expression from the model, returns a lambda function that
		 * computes the derivative of the expression starting from the data provided
		 * by IDA, the value of all induction variables and the variable with
		 * respect to which the expression must be differentiated.
		 */
		[[nodiscard]] Function getDerFunction(const model::Expression &exp);

		/**
		 * IDAResFn user-defined residual function, passed to IDA through IDAInit.
		 * It contains how to compute the Residual function of the system, starting
		 * from the provided UserData struct, iterating through every equation.
		 */
		static int residualFunction(
				realtype tt, N_Vector yy, N_Vector yp, N_Vector rr, void *user_data);

		/**
		 * IDALsJacFn user-defined Jacobian approximation function, passed to IDA
		 * through IDASetJacFn. It contains how to compute the Jacobian Matrix of
		 * the system, starting from the provided UserData struct, iterating through
		 * every equation and variable. The matrix is represented in CSR format.
		 */
		static int jacobianMatrix(
				realtype tt,
				realtype cj,
				N_Vector yy,
				N_Vector yp,
				N_Vector rr,
				SUNMatrix JJ,
				void *user_data,
				N_Vector tempv1,
				N_Vector tempv2,
				N_Vector tempv3);

		/**
		 * Check an IDA function return value in order to find possible failures.
		 */
		[[nodiscard]] static mlir::LogicalResult checkRetval(
				void *retval, const char *funcname, int opt);

		private:
		// Model data
		model::Model model;
		std::map<model::Variable, realtype> initialValueMap;
		std::map<model::Variable, size_t> indexOffsetMap;
		UserData *data;

		// Simulation times
		const realtype startTime;
		const realtype stopTime;
		realtype time;

		// Error tolerances
		const realtype relativeTolerance;
		const realtype absoluteTolerance;

		// Matrix size
		const sunindextype equationsNumber;
		const sunindextype nonZeroValuesNumber;

		// Variables vectors and values
		N_Vector variablesVector;
		N_Vector derivativesVector;
		N_Vector idVector;
		realtype *variablesValues;
		realtype *derivativesValues;
		realtype *idValues;

		// IDA classes
		void *idaMemory;
		int retval;
		SUNMatrix sparseMatrix;
		SUNLinearSolver linearSolver;
		SUNNonlinearSolver nonlinearSolver;
	};
}	 // namespace modelica::codegen::ida
