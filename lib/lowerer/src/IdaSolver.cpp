#include "modelica/lowerer/IdaSolver.hpp"

#include "modelica/model/ModErrors.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

IdaSolver::IdaSolver(
		const ModBltBlock &bltBlock,
		const realtype start_time,
		const realtype stop_time,
		const realtype step_size,
		const realtype reltol,
		const realtype abstol)
		: bltBlock(bltBlock),
			start_time(start_time),
			stop_time(stop_time),
			step_size(step_size),
			reltol(reltol),
			abstol(abstol),
			neq(bltBlock.size()),
			nnz(neq * neq)
{
}

IdaSolver::~IdaSolver()
{
	// Free memory
	IDAFree(&ida_mem);
	SUNLinSolFree(LS);
	SUNMatDestroy(A);
	N_VDestroy(yy);
	N_VDestroy(yp);
}

Error IdaSolver::init()
{
	// Allocate N-vectors.
	yy = N_VNew_Serial(neq);
	if (auto error = check_retval((void *) yy, "N_VNew_Serial", 0); error)
		return move(error);
	yp = N_VNew_Serial(neq);
	if (auto error = check_retval((void *) yp, "N_VNew_Serial", 0); error)
		return move(error);

	// Create and initialize  y, y'
	yval = N_VGetArrayPointer(yy);
	yval[0] = RCONST(0.0);
	yval[1] = RCONST(0.0);
	yval[2] = RCONST(0.0);

	ypval = N_VGetArrayPointer(yp);
	ypval[0] = RCONST(-0.04);
	ypval[1] = RCONST(0.04);
	ypval[2] = RCONST(0.0);

	// Call IDACreate and IDAInit to initialize IDA memory
	ida_mem = IDACreate();
	if (auto error = check_retval((void *) ida_mem, "IDACreate", 0); error)
		return move(error);

	retval = IDASetUserData(ida_mem, (void *) this);
	if (auto error = check_retval(&retval, "IDASetUserData", 1); error)
		return move(error);

	retval = IDAInit(ida_mem, IdaSolver::resrob, start_time, yy, yp);
	if (auto error = check_retval(&retval, "IDAInit", 1); error)
		return move(error);

	// Call IDASVtolerances to set tolerances
	retval = IDASStolerances(ida_mem, reltol, abstol);
	if (auto error = check_retval(&retval, "IDASVtolerances", 1); error)
		return move(error);

	// Create sparse SUNMatrix for use in linear solves
	A = SUNSparseMatrix(neq, neq, nnz, CSR_MAT);
	if (auto error = check_retval((void *) A, "SUNSparseMatrix", 0); error)
		return move(error);

	// Create KLU SUNLinearSolver object
	LS = SUNLinSol_KLU(yy, A);
	if (auto error = check_retval((void *) LS, "SUNLinSol_KLU", 0); error)
		return move(error);

	// Attach the matrix and linear solver
	retval = IDASetLinearSolver(ida_mem, LS, A);
	if (auto error = check_retval(&retval, "IDASetLinearSolver", 1); error)
		return move(error);

	// Set the user-supplied Jacobian routine
	retval = IDASetJacFn(ida_mem, IdaSolver::jacrob);
	if (auto error = check_retval(&retval, "IDASetJacFn", 1); error)
		return move(error);

	return Error::success();
}

Expected<bool> IdaSolver::step()
{
	time += step_size;

	retval = IDASolve(ida_mem, time, &tret, yy, yp, IDA_NORMAL);

	if (auto error = check_retval(&retval, "IDASolve", 1); error)
		return move(error);

	return time < stop_time;
}

int IdaSolver::resrob(
		realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void *user_data)
{
	realtype *yval = N_VGetArrayPointer(yy);
	realtype *ypval = N_VGetArrayPointer(yp);
	realtype *rval = N_VGetArrayPointer(rr);

	IdaSolver *idaSolver = static_cast<IdaSolver *>(user_data);

	// TODO: Copmute the Residual Function.

	return 0;
}

int IdaSolver::jacrob(
		realtype tt,
		realtype cj,
		N_Vector yy,
		N_Vector yp,
		N_Vector resvec,
		SUNMatrix JJ,
		void *user_data,
		N_Vector tempv1,
		N_Vector tempv2,
		N_Vector tempv3)
{
	realtype *yval = N_VGetArrayPointer(yy);
	sunindextype *rowptrs = SUNSparseMatrix_IndexPointers(JJ);
	sunindextype *colvals = SUNSparseMatrix_IndexValues(JJ);
	realtype *data = SUNSparseMatrix_Data(JJ);
	SUNMatZero(JJ);

	IdaSolver *idaSolver = static_cast<IdaSolver *>(user_data);

	for (int i = 0; i <= idaSolver->neq; ++i)
	{
		rowptrs[i] = idaSolver->neq * i;
	}

	//  TODO: Compute the Jacobian Matrix.

	return 0;
}

Error IdaSolver::check_retval(void *returnvalue, const char *funcname, int opt)
{
	// Check if SUNDIALS function returned NULL pointer (no memory allocated)
	if (opt == 0 && returnvalue == NULL)
		return make_error<SundialsError>(returnvalue, funcname, opt);

	// Check if SUNDIALS function returned a positive integer value
	if (opt == 1 && *((int *) returnvalue) < 0)
		return make_error<SundialsError>(returnvalue, funcname, opt);

	// Check if function returned NULL pointer (no memory allocated)
	if (opt == 2 && returnvalue == NULL)
		return make_error<SundialsError>(returnvalue, funcname, opt);

	return Error::success();
}
