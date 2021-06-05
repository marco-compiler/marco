#pragma once

#include <ida/ida.h>
#include <math.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_klu.h>
#include <sunmatrix/sunmatrix_sparse.h>

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
				const ModBltBlock &bltBlock,
				const realtype start_time,
				const realtype stop_time,
				const realtype step_size,
				const realtype reltol,
				const realtype abstol);

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
		 * This function is called by the IDA solver. It contains how to compute the
		 * Residual Function of the BLT block given the value of the current
		 * parameters.
		 */
		static int resrob(
				realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void *user_data);

		/**
		 * This function is called by the IDA solver. It contains how to compute the
		 * Jacobian Matrix of the BLT block given the value of the current
		 * parameters.
		 */
		static int jacrob(
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
		llvm::Error check_retval(void *returnvalue, const char *funcname, int opt);

		const ModBltBlock &bltBlock;	// BLT block solved by IDA.

		void *ida_mem;	// Pointer to the ida memory block.
		int retval;			// Return value of IDA functions, used to check for errors.

		const realtype start_time;	// Start time of the integration.
		const realtype stop_time;		// Stop time of the integration.
		const realtype step_size;		// Step size of the integration.

		const realtype reltol;	// Relative tolerance.
		const realtype abstol;	// Absolute tolerance.

		const sunindextype neq;	 // Number of equations.
		const sunindextype nnz;	 // Maximum number of non-zero values in the matrix.

		realtype time;	// Current time of the integration.
		realtype tret;	// Time reached by the solver at each step.

		N_Vector yy;	// Vectors of the variables.
		N_Vector yp;	// Vectors of the variable derivatives.

		realtype *yval;		// Vectors of the variable values.
		realtype *ypval;	// Vectors of the variable derivative values.

		SUNMatrix A;				 // Sparse matrix.
		SUNLinearSolver LS;	 // Linear solver.
	};
}	 // namespace modelica
