#pragma once

#include <ida/ida.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_klu.h>
#include <sunmatrix/sunmatrix_sparse.h>

#include "llvm/Support/Error.h"
#include "modelica/lowerer/LowererUtils.hpp"
#include "modelica/model/ModBltBlock.hpp"

namespace modelica
{
	/**
	 * This class interfaces with the SUNDIALS IDA library in order to solve a
	 * BLT block, which can contain an Algebraic loop, an Implicit equation or a
	 * Differential equation.
	 */
	class IdaSolver
	{
		public:
		/**
		 * The constructor initializes the class with a ModBltBlock and some other
		 * parameters.
		 */
		IdaSolver(
				LowererContext &info,
				const llvm::SmallVector<ModBltBlock, 3> &bltBlocks,
				const realtype startTime = 0.0,
				const realtype stopTime = 1.0,
				const realtype relativeTolerance = 1e-6,
				const realtype abstol = 1e-6);

		/**
		 * The destructor frees the memory used by the SUNDIALS library.
		 */
		~IdaSolver();

		/**
		 * This function creates the necessary classes used by the SUNDIALS library.
		 * It must be called before the first usage of step(). It may fail in case
		 * of malformed model.
		 */
		llvm::Error init();

		/**
		 * This function perform a step of computation in solving the given
		 * ModBltBlock. Notice that SUNDIALS IDA uses a variable step size control.
		 */
		llvm::Expected<bool> step();

		private:
		/**
		 * Compute the number of equations in the sparse matrix.
		 */
		sunindextype computeNEQ();

		/**
		 * Compute the number of non-zero values in the sparse matrix.
		 */
		sunindextype computeNNZ();

		/**
		 * Initialize the variables and derivatives vectors, which contain the
		 * values of all the variables and their derivatives values, and the id
		 * vector, which contains if the variables are algebraic or differential.
		 */
		void initVectors();

		/**
		 * This function is called by the IDA solver. It contains how to compute the
		 * Residual Function of the BLT block given the value of the current
		 * parameters.
		 */
		static int residualFunction(
				realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void *user_data);

		/**
		 * This function is called by the IDA solver. It contains how to compute the
		 * Jacobian Matrix of the BLT block given the value of the current
		 * parameters. The matrix is represented in Compressed Sparse Row format.
		 */
		static int jacobianMatrix(
				realtype tt,
				realtype cj,
				N_Vector yy,
				N_Vector yp,
				N_Vector resvec,
				SUNMatrix JJ,
				void *user_data,
				N_Vector tempv1,
				N_Vector tempv2,
				N_Vector tempv3);

		/**
		 * Check a function return value in order to find possible failures.
		 */
		llvm::Error checkRetval(void *returnvalue, const char *funcname, int opt);

		// Lowerer data
		const LowererContext &context;
		const llvm::SmallVector<ModBltBlock, 3> &bltBlocks;

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
		int returnValue;
		SUNMatrix sparseMatrix;
		SUNLinearSolver linearSolver;
	};
}	 // namespace modelica
