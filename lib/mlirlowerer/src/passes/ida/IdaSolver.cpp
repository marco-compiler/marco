#include <modelica/mlirlowerer/passes/ida/IdaSolver.h>
#include <modelica/mlirlowerer/passes/model/BltBlock.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>
#include <modelica/utils/Interval.hpp>

#define exitOnError(f)                                                         \
	if (auto error = f; failed(error))                                           \
		return error;

using namespace modelica::codegen::ida;
using namespace modelica::codegen::model;

static double getValue(modelica::codegen::ConstantOp constantOp)
{
	mlir::Attribute attribute = constantOp.value();

	if (auto integer = attribute.dyn_cast<modelica::codegen::IntegerAttribute>())
		return integer.getValue();

	if (auto real = attribute.dyn_cast<modelica::codegen::RealAttribute>())
		return real.getValue();

	assert(false && "Unreachable");
	return 0.0;
}

IdaSolver::IdaSolver(
		model::Model &model,
		const realtype startTime,
		const realtype stopTime,
		const realtype relativeTolerance,
		const realtype absoluteTolerance)
		: model(model),
			startTime(startTime),
			stopTime(stopTime),
			relativeTolerance(relativeTolerance),
			absoluteTolerance(absoluteTolerance),
			equationsNumber(computeNEQ()),
			nonZeroValuesNumber(computeNNZ())
{
}

mlir::LogicalResult IdaSolver::init()
{
	// Create and initialize the required N-vectors for the variables.
	variablesVector = N_VNew_Serial(equationsNumber);
	exitOnError(checkRetval((void *) variablesVector, "N_VNew_Serial", 0));

	derivativesVector = N_VNew_Serial(equationsNumber);
	exitOnError(checkRetval((void *) derivativesVector, "N_VNew_Serial", 0));

	idVector = N_VNew_Serial(equationsNumber);
	exitOnError(checkRetval((void *) idVector, "N_VNew_Serial", 0));

	initData();

	// Initialize IDA memory.
	idaMemory = IDACreate();
	exitOnError(checkRetval((void *) idaMemory, "IDACreate", 0));

	retval = IDASetUserData(idaMemory, (void *) data);
	exitOnError(checkRetval(&retval, "IDASetUserData", 1));

	retval = IDASetId(idaMemory, idVector);
	exitOnError(checkRetval(&retval, "IDASetId", 1));

	retval = IDASetStopTime(idaMemory, stopTime);
	exitOnError(checkRetval(&retval, "IDASetStopTime", 1));

	retval = IDAInit(
			idaMemory,
			IdaSolver::residualFunction,
			startTime,
			variablesVector,
			derivativesVector);
	exitOnError(checkRetval(&retval, "IDAInit", 1));

	// Call IDASStolerances to set tolerances.
	retval = IDASStolerances(idaMemory, relativeTolerance, absoluteTolerance);
	exitOnError(checkRetval(&retval, "IDASStolerances", 1));

	// Create sparse SUNMatrix for use in linear solver.
	sparseMatrix = SUNSparseMatrix(
			equationsNumber, equationsNumber, nonZeroValuesNumber, CSR_MAT);
	exitOnError(checkRetval((void *) sparseMatrix, "SUNSparseMatrix", 0));

	// Create and attach a KLU SUNLinearSolver object.
	linearSolver = SUNLinSol_KLU(variablesVector, sparseMatrix);
	exitOnError(checkRetval((void *) linearSolver, "SUNLinSol_KLU", 0));

	retval = IDASetLinearSolver(idaMemory, linearSolver, sparseMatrix);
	exitOnError(checkRetval(&retval, "IDASetLinearSolver", 1));

	// Create and attach a Newton NonlinearSolver object.
	nonlinearSolver = SUNNonlinSol_Newton(variablesVector);
	exitOnError(checkRetval((void *) nonlinearSolver, "SUNNonlinSol_Newton", 0));

	retval = IDASetNonlinearSolver(idaMemory, nonlinearSolver);
	exitOnError(checkRetval(&retval, "IDASetNonlinearSolver", 1));

	// Set the user-supplied Jacobian routine.
	retval = IDASetJacFn(idaMemory, IdaSolver::jacobianMatrix);
	exitOnError(checkRetval(&retval, "IDASetJacFn", 1));

	// Call IDACalcIC to correct the initial values.
	retval = IDACalcIC(idaMemory, IDA_YA_YDP_INIT, stopTime);
	exitOnError(checkRetval(&retval, "IDACalcIC", 1));

	return mlir::LogicalResult::success();
}

std::optional<bool> IdaSolver::step()
{
	// Execute one step
	retval = IDASolve(
			idaMemory,
			stopTime,
			&time,
			variablesVector,
			derivativesVector,
			IDA_ONE_STEP);

	// Check if the solver failed
	if (failed(checkRetval(&retval, "IDASolve", 1)))
		return std::nullopt;

	// Return if the computation has not reached the stop time yet.
	return time < stopTime;
}

mlir::LogicalResult IdaSolver::run(llvm::raw_ostream &OS)
{
	while (true)
	{
		auto result = step();

		if (!result)
			return mlir::failure();

		printOutput(OS);

		if (!*result)
		{
			exitOnError(printStats(OS));

			return mlir::success();
		}
	}
}

mlir::LogicalResult IdaSolver::free()
{
	// Free memory
	IDAFree(&idaMemory);
	retval = SUNNonlinSolFree(nonlinearSolver);
	exitOnError(checkRetval(&retval, "SUNNonlinSolFree", 1));
	retval = SUNLinSolFree(linearSolver);
	exitOnError(checkRetval(&retval, "SUNLinSolFree", 1));
	SUNMatDestroy(sparseMatrix);
	N_VDestroy(variablesVector);
	N_VDestroy(derivativesVector);
	N_VDestroy(idVector);
	delete data;

	return mlir::success();
}

void IdaSolver::printOutput(llvm::raw_ostream &OS)
{
	realtype *yval = N_VGetArrayPointer(variablesVector);

	OS << time;

	for (size_t i : irange(equationsNumber))
		OS << ", " << yval[i];

	OS << "\n";
}

mlir::LogicalResult IdaSolver::printStats(llvm::raw_ostream &OS)
{
	long int nst, nni, nje, nre, netf, ncfn;

	retval = IDAGetNumSteps(idaMemory, &nst);
	exitOnError(checkRetval(&retval, "IDAGetNumSteps", 1));

	retval = IDAGetNumResEvals(idaMemory, &nre);
	exitOnError(checkRetval(&retval, "IDAGetNumResEvals", 1));

	retval = IDAGetNumJacEvals(idaMemory, &nje);
	exitOnError(checkRetval(&retval, "IDAGetNumJacEvals", 1));

	retval = IDAGetNumNonlinSolvIters(idaMemory, &nni);
	exitOnError(checkRetval(&retval, "IDAGetNumNonlinSolvIters", 1));

	retval = IDAGetNumErrTestFails(idaMemory, &netf);
	exitOnError(checkRetval(&retval, "IDAGetNumErrTestFails", 1));

	retval = IDAGetNumNonlinSolvConvFails(idaMemory, &ncfn);
	exitOnError(checkRetval(&retval, "IDAGetNumNonlinSolvConvFails", 1));

	OS << "\nFinal Run Statistics: \n\n";
	OS << "Number of steps                    = " << nst << "\n";
	OS << "Number of residual evaluations     = " << nre << "\n";
	OS << "Number of Jacobian evaluations     = " << nje << "\n";
	OS << "Number of nonlinear iterations     = " << nni << "\n";
	OS << "Number of error test failures      = " << netf << "\n";
	OS << "Number of nonlinear conv. failures = " << ncfn << "\n";

	return mlir::success();
}

sunindextype IdaSolver::computeNEQ()
{
	sunindextype result = 0;

	for (BltBlock &bltBlock : model.getBltBlocks())
		result += bltBlock.size();

	return result;
}

sunindextype IdaSolver::computeNNZ()
{
	sunindextype result = 0, rowLength = 0;
	std::set<model::Variable> varSet;

	for (BltBlock &bltBlock : model.getBltBlocks())
	{
		for (Equation &equation : bltBlock.getEquations())
		{
			Variable var =
					model.getVariable(equation.getDeterminedVariable().getVar());

			if (varSet.find(var) == varSet.end())
			{
				varSet.insert(var);
				rowLength += var.toMultiDimInterval().size();
			}
		}

		result += rowLength * bltBlock.size();
	}

	return result;
}

void IdaSolver::initData()
{
	variablesValues = N_VGetArrayPointer(variablesVector);
	derivativesValues = N_VGetArrayPointer(derivativesVector);
	idValues = N_VGetArrayPointer(idVector);

	// Create and load problem data block.
	data = new UserData;

	size_t rowLength = 0;

	// TODO: Add different value handling for initialization

	// Map all vector variables to their initial value.
	model.getOp().init().walk([&](FillOp fillOp) {
		if (!model.hasVariable(fillOp.memory()))
			return;

		Variable var = model.getVariable(fillOp.memory());

		mlir::Operation *op = fillOp.value().getDefiningOp();
		ConstantOp constantOp = mlir::dyn_cast<ConstantOp>(op);
		realtype value = getValue(constantOp);

		initialValueMap[var] = value;
		assert(!var.isDerivative());
		if (var.isState())
			initialValueMap[model.getVariable(var.getDer())] = value;
	});

	// Map all scalar variables to their initial value.
	model.getOp().init().walk([&](AssignmentOp assignmentOp) {
		mlir::Operation *op = assignmentOp.destination().getDefiningOp();
		SubscriptionOp subscriptionOp = mlir::dyn_cast<SubscriptionOp>(op);

		if (!model.hasVariable(subscriptionOp.source()))
			return;

		Variable var = model.getVariable(subscriptionOp.source());

		op = assignmentOp.source().getDefiningOp();
		ConstantOp constantOp = mlir::dyn_cast<ConstantOp>(op);
		realtype value = getValue(constantOp);

		initialValueMap[var] = value;
		assert(!var.isDerivative());
		if (var.isState())
			initialValueMap[model.getVariable(var.getDer())] = value;
	});

	for (BltBlock &bltBlock : model.getBltBlocks())
	{
		for (Equation &equation : bltBlock.getEquations())
		{
			// Get the variable matched with every equation.
			Variable var =
					model.getVariable(equation.getDeterminedVariable().getVar());

			assert(!var.isTrivial());

			// If the variable has not been insterted yet, initialize it.
			if (indexOffsetMap.find(var) == indexOffsetMap.end())
			{
				// Note the variable offset from the beginning of the variable array.
				indexOffsetMap[var] = rowLength;

				if (var.isState())
					indexOffsetMap[model.getVariable(var.getDer())] = rowLength;
				else if (var.isDerivative())
					indexOffsetMap[model.getVariable(var.getState())] = rowLength;

				// Initialize variablesValues, derivativesValues, idValues.
				for (size_t i : irange(var.toMultiDimInterval().size()))
				{
					variablesValues[rowLength + i] = initialValueMap[var];
					derivativesValues[rowLength + i] = 0.0;
					idValues[rowLength + i] =
							(var.isState() || var.isDerivative()) ? 1.0 : 0.0;
				}

				// Increase the length of the current row.
				rowLength += var.toMultiDimInterval().size();
			}
		}

		// Initialize UserData with all parameters needed by IDA.
		for (Equation &equation : bltBlock.getEquations())
		{
			data->rowLength.push_back(rowLength);
			data->dimensions.push_back(getDimensions(equation));
			data->residuals.push_back(getResidual(equation));
			data->jacobianMatrix.push_back(getJacobian(equation));
		}
	}

	assert((sunindextype) rowLength == equationsNumber);
}

Dimensions IdaSolver::getDimensions(const Equation &equation)
{
	Dimensions dimension;

	for (modelica::Interval &interval : equation.getInductions())
		dimension.push_back({ interval.min() - 1, interval.max() - 1 });

	return dimension;
}

Function IdaSolver::getResidual(const Equation &equation)
{
	// Return a lambda function that subtract from the right side of the equation,
	// the left side of the equation.
	Function left = getFunction(equation.lhs());
	Function right = getFunction(equation.rhs());

	return [left, right](
						 double tt,
						 double cj,
						 double *yy,
						 double *yp,
						 Indexes &ind,
						 double var) -> double {
		return right(tt, cj, yy, yp, ind, var) - left(tt, cj, yy, yp, ind, var);
	};
}

Function IdaSolver::getJacobian(const Equation &equation)
{
	// Return a lambda function that subtract from the derivative of the right
	// side of the equation, the derivative of the left side of the equation.
	Function left = getDerFunction(equation.lhs());
	Function right = getDerFunction(equation.rhs());

	return [left, right](
						 double tt,
						 double cj,
						 double *yy,
						 double *yp,
						 Indexes &ind,
						 size_t var) -> double {
		return right(tt, cj, yy, yp, ind, var) - left(tt, cj, yy, yp, ind, var);
	};
}

Function IdaSolver::getFunction(const Expression &expression)
{
	mlir::Operation *definingOp = expression.getOp();

	// Constant value.
	if (auto op = mlir::dyn_cast<ConstantOp>(definingOp))
	{
		realtype value = getValue(op);

		return [value](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double { return value; };
	}

	// Scalar variable reference.
	if (expression.isReference())
	{
		// Time variable
		Variable var = model.getVariable(expression.getReferredVectorAccess());
		if (indexOffsetMap.find(var) == indexOffsetMap.end())
			return [](double tt,
								double cj,
								double *yy,
								double *yp,
								Indexes &ind,
								double var) -> double { return tt; };

		size_t offset = indexOffsetMap[var];

		if (var.isDerivative())
			return [offset](
								 double tt,
								 double cj,
								 double *yy,
								 double *yp,
								 Indexes &ind,
								 double var) -> double { return yp[offset]; };
		else
			return [offset](
								 double tt,
								 double cj,
								 double *yy,
								 double *yp,
								 Indexes &ind,
								 double var) -> double { return yy[offset]; };
	}

	assert(expression.isOperation());

	// Vector variable reference.
	if (expression.isReferenceAccess())
	{
		// Compute the IDA offset of the variable in the 1D array variablesVector.
		Variable var = model.getVariable(expression.getReferredVectorAccess());
		assert(indexOffsetMap.find(var) != indexOffsetMap.end());
		size_t offset = indexOffsetMap[var];

		// Compute the access offset based on the induction variables of the
		// for-equation.
		VectorAccess vectorAccess = AccessToVar::fromExp(expression).getAccess();
		std::vector<std::pair<sunindextype, sunindextype>> access;

		for (auto &acc : vectorAccess.getMappingOffset())
		{
			size_t accOffset =
					acc.isDirectAccess() ? acc.getOffset() : acc.getOffset() + 1;
			size_t accInduction = acc.isOffset() ? acc.getInductionVar() : -1;
			access.push_back({ accOffset, accInduction });
		}

		// Compute the multi-dimensional offset of the array.
		modelica::MultiDimInterval dimensions = var.toMultiDimInterval();
		std::vector<size_t> dim;
		for (size_t i = 1; i < dimensions.dimensions(); i++)
		{
			for (size_t j = 0; j < dim.size(); j++)
				dim[j] *= dimensions.at(i).size();
			dim.push_back(dimensions.at(i).size());
		}
		dim.push_back(1);

		assert(access.size() == dim.size());

		if (var.isDerivative())
			return [offset, access, dim](
								 double tt,
								 double cj,
								 double *yy,
								 double *yp,
								 Indexes &ind,
								 double var) -> double {
				size_t varOffset = 0;

				for (size_t i = 0; i < access.size(); i++)
				{
					auto acc = access[i];
					size_t accOffset =
							acc.first + (acc.second != -1 ? ind[acc.second] : 0);
					varOffset += accOffset * dim[i];
				}

				return yp[offset + varOffset];
			};
		else
			return [offset, access, dim](
								 double tt,
								 double cj,
								 double *yy,
								 double *yp,
								 Indexes &ind,
								 double var) -> double {
				size_t varOffset = 0;

				for (size_t i = 0; i < access.size(); i++)
				{
					auto acc = access[i];
					size_t accOffset =
							acc.first + (acc.second != -1 ? ind[acc.second] : 0);
					varOffset += accOffset * dim[i];
				}

				return yy[offset + varOffset];
			};
	}

	// Get the lambda functions to compute the values of all the children.
	std::vector<Function> children;
	for (size_t i : irange(expression.childrenCount()))
		children.push_back(getFunction(expression.getChild(i)));

	if (mlir::isa<NegateOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return -operand(tt, cj, yy, yp, ind, var);
		};

	if (mlir::isa<AddOp>(definingOp))
		return [left = children[0], right = children[1]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return left(tt, cj, yy, yp, ind, var) + right(tt, cj, yy, yp, ind, var);
		};

	if (mlir::isa<SubOp>(definingOp))
		return [left = children[0], right = children[1]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return left(tt, cj, yy, yp, ind, var) - right(tt, cj, yy, yp, ind, var);
		};

	if (mlir::isa<MulOp>(definingOp))
		return [left = children[0], right = children[1]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return left(tt, cj, yy, yp, ind, var) * right(tt, cj, yy, yp, ind, var);
		};

	if (mlir::isa<DivOp>(definingOp))
		return [left = children[0], right = children[1]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return left(tt, cj, yy, yp, ind, var) / right(tt, cj, yy, yp, ind, var);
		};

	if (mlir::isa<PowOp>(definingOp))
		return [base = children[0], exponent = children[1]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::pow(
					base(tt, cj, yy, yp, ind, var), exponent(tt, cj, yy, yp, ind, var));
		};

	if (mlir::isa<AbsOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::abs(operand(tt, cj, yy, yp, ind, var));
		};

	if (mlir::isa<SignOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			double x = operand(tt, cj, yy, yp, ind, var);

			return (x > 0.0) - (x < 0.0);
		};

	if (mlir::isa<SqrtOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::sqrt(operand(tt, cj, yy, yp, ind, var));
		};

	if (mlir::isa<ExpOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::exp(operand(tt, cj, yy, yp, ind, var));
		};

	if (mlir::isa<LogOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::log(operand(tt, cj, yy, yp, ind, var));
		};

	if (mlir::isa<Log10Op>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::log10(operand(tt, cj, yy, yp, ind, var));
		};

	if (mlir::isa<SinOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::sin(operand(tt, cj, yy, yp, ind, var));
		};

	if (mlir::isa<CosOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::cos(operand(tt, cj, yy, yp, ind, var));
		};

	if (mlir::isa<TanOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::tan(operand(tt, cj, yy, yp, ind, var));
		};

	if (mlir::isa<AsinOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::asin(operand(tt, cj, yy, yp, ind, var));
		};

	if (mlir::isa<AcosOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::acos(operand(tt, cj, yy, yp, ind, var));
		};

	if (mlir::isa<AtanOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::atan(operand(tt, cj, yy, yp, ind, var));
		};

	if (mlir::isa<Atan2Op>(definingOp))
		return [y = children[0], x = children[1]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::atan2(
					y(tt, cj, yy, yp, ind, var), x(tt, cj, yy, yp, ind, var));
		};

	if (mlir::isa<SinhOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::sinh(operand(tt, cj, yy, yp, ind, var));
		};

	if (mlir::isa<CoshOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::cosh(operand(tt, cj, yy, yp, ind, var));
		};

	if (mlir::isa<TanhOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::tanh(operand(tt, cj, yy, yp, ind, var));
		};

	// TODO: Handle CallOp

	assert(false && "Unexpected operation");
}

Function IdaSolver::getDerFunction(const Expression &expression)
{
	mlir::Operation *definingOp = expression.getOp();

	// Constant value.
	if (mlir::isa<ConstantOp>(definingOp))
	{
		return [](double tt,
							double cj,
							double *yy,
							double *yp,
							Indexes &ind,
							double var) -> double { return 0.0; };
	}

	// Scalar variable reference.
	if (expression.isReference())
	{
		// Time variable
		Variable var = model.getVariable(expression.getReferredVectorAccess());
		if (indexOffsetMap.find(var) == indexOffsetMap.end())
			return [](double tt,
								double cj,
								double *yy,
								double *yp,
								Indexes &ind,
								double var) -> double { return 0.0; };

		size_t offset = indexOffsetMap[var];

		if (var.isDerivative())
			return [offset](
								 double tt,
								 double cj,
								 double *yy,
								 double *yp,
								 Indexes &ind,
								 double var) -> double {
				if (offset == var)
					return cj;
				return 0.0;
			};
		else
			return [offset](
								 double tt,
								 double cj,
								 double *yy,
								 double *yp,
								 Indexes &ind,
								 double var) -> double {
				if (offset == var)
					return 1.0;
				return 0.0;
			};
	}

	assert(expression.isOperation());

	// Vector variable reference.
	if (expression.isReferenceAccess())
	{
		// Compute the IDA offset of the variable in the 1D array variablesVector.
		Variable var = model.getVariable(expression.getReferredVectorAccess());
		assert(indexOffsetMap.find(var) != indexOffsetMap.end());
		size_t offset = indexOffsetMap[var];

		// Compute the access offset based on the induction variables of the
		// for-equation.
		VectorAccess vectorAccess = AccessToVar::fromExp(expression).getAccess();
		std::vector<std::pair<sunindextype, sunindextype>> access;

		for (auto &acc : vectorAccess.getMappingOffset())
		{
			size_t accOffset =
					acc.isDirectAccess() ? acc.getOffset() : acc.getOffset() + 1;
			size_t accInduction = acc.isOffset() ? acc.getInductionVar() : -1;
			access.push_back({ accOffset, accInduction });
		}

		// Compute the multi-dimensional offset of the array.
		modelica::MultiDimInterval dimensions = var.toMultiDimInterval();
		std::vector<size_t> dim;
		for (size_t i = 1; i < dimensions.dimensions(); i++)
		{
			for (size_t j = 0; j < dim.size(); j++)
				dim[j] *= dimensions.at(i).size();
			dim.push_back(dimensions.at(i).size());
		}
		dim.push_back(1);

		assert(access.size() == dim.size());

		if (var.isDerivative())
			return [offset, access, dim](
								 double tt,
								 double cj,
								 double *yy,
								 double *yp,
								 Indexes &ind,
								 double var) -> double {
				size_t varOffset = 0;

				for (size_t i = 0; i < access.size(); i++)
				{
					auto acc = access[i];
					size_t accOffset =
							acc.first + (acc.second != -1 ? ind[acc.second] : 0);
					varOffset += accOffset * dim[i];
				}

				if (offset + varOffset == var)
					return cj;
				return 0.0;
			};
		else
			return [offset, access, dim](
								 double tt,
								 double cj,
								 double *yy,
								 double *yp,
								 Indexes &ind,
								 double var) -> double {
				size_t varOffset = 0;

				for (size_t i = 0; i < access.size(); i++)
				{
					auto acc = access[i];
					size_t accOffset =
							acc.first + (acc.second != -1 ? ind[acc.second] : 0);
					varOffset += accOffset * dim[i];
				}

				if (offset + varOffset == var)
					return 1.0;
				return 0.0;
			};
	}

	if (mlir::isa<SignOp>(definingOp))
		return [](double tt,
							double cj,
							double *yy,
							double *yp,
							Indexes &ind,
							double var) -> double { return 0.0; };

	// Get the lambda functions to compute the values of the derivatives of all
	// the children.
	std::vector<Function> derChildren;
	for (size_t i : irange(expression.childrenCount()))
		derChildren.push_back(getDerFunction(expression.getChild(i)));

	if (mlir::isa<AddOp>(definingOp))
		return [derLeft = derChildren[0], derRight = derChildren[1]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return derLeft(tt, cj, yy, yp, ind, var) +
						 derRight(tt, cj, yy, yp, ind, var);
		};

	if (mlir::isa<SubOp>(definingOp))
		return [derLeft = derChildren[0], derRight = derChildren[1]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return derLeft(tt, cj, yy, yp, ind, var) -
						 derRight(tt, cj, yy, yp, ind, var);
		};

	if (mlir::isa<NegateOp>(definingOp))
		return [derOperand = derChildren[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return -derOperand(tt, cj, yy, yp, ind, var);
		};

	// Get the lambda functions to compute the values of all the children.
	std::vector<Function> children;
	for (size_t i : irange(expression.childrenCount()))
		children.push_back(getFunction(expression.getChild(i)));

	if (mlir::isa<MulOp>(definingOp))
		return [left = children[0],
						right = children[1],
						derLeft = derChildren[0],
						derRight = derChildren[1]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return left(tt, cj, yy, yp, ind, var) *
								 derRight(tt, cj, yy, yp, ind, var) +
						 right(tt, cj, yy, yp, ind, var) *
								 derLeft(tt, cj, yy, yp, ind, var);
		};

	if (mlir::isa<DivOp>(definingOp))
		return [left = children[0],
						right = children[1],
						derLeft = derChildren[0],
						derRight = derChildren[1]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			double rightValue = right(tt, cj, yy, yp, ind, var);
			double dividend =
					rightValue * derLeft(tt, cj, yy, yp, ind, var) -
					left(tt, cj, yy, yp, ind, var) * derRight(tt, cj, yy, yp, ind, var);
			return dividend / (rightValue * rightValue);
		};

	if (mlir::isa<PowOp>(definingOp))
		return
				[base = children[0], exponent = children[1], derBase = derChildren[0]](
						double tt,
						double cj,
						double *yy,
						double *yp,
						Indexes &ind,
						double var) -> double {
					double exponentValue = exponent(tt, cj, yy, yp, ind, var);
					return exponentValue *
								 std::pow(base(tt, cj, yy, yp, ind, var), exponentValue) *
								 derBase(tt, cj, yy, yp, ind, var);
				};

	if (mlir::isa<AbsOp>(definingOp))
		return [operand = children[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			double x = operand(tt, cj, yy, yp, ind, var);

			return (x > 0.0) - (x < 0.0);
		};

	if (mlir::isa<SqrtOp>(definingOp))
		return [operand = children[0], derOperand = derChildren[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return derOperand(tt, cj, yy, yp, ind, var) /
						 std::sqrt(operand(tt, cj, yy, yp, ind, var)) / 2;
		};

	if (mlir::isa<ExpOp>(definingOp))
		return [operand = children[0], derOperand = derChildren[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::exp(operand(tt, cj, yy, yp, ind, var)) *
						 derOperand(tt, cj, yy, yp, ind, var);
		};

	if (mlir::isa<LogOp>(definingOp))
		return [operand = children[0], derOperand = derChildren[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return derOperand(tt, cj, yy, yp, ind, var) /
						 operand(tt, cj, yy, yp, ind, var);
		};

	if (mlir::isa<Log10Op>(definingOp))
		return [operand = children[0], derOperand = derChildren[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return derOperand(tt, cj, yy, yp, ind, var) /
						 operand(tt, cj, yy, yp, ind, var) / std::log(10);
		};

	if (mlir::isa<SinOp>(definingOp))
		return [operand = children[0], derOperand = derChildren[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::cos(operand(tt, cj, yy, yp, ind, var)) *
						 derOperand(tt, cj, yy, yp, ind, var);
		};

	if (mlir::isa<CosOp>(definingOp))
		return [operand = children[0], derOperand = derChildren[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return -std::sin(operand(tt, cj, yy, yp, ind, var)) *
						 derOperand(tt, cj, yy, yp, ind, var);
		};

	if (mlir::isa<TanOp>(definingOp))
		return [operand = children[0], derOperand = derChildren[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			double tanOperandValue = std::tan(operand(tt, cj, yy, yp, ind, var));
			return (1 + tanOperandValue * tanOperandValue) *
						 derOperand(tt, cj, yy, yp, ind, var);
		};

	if (mlir::isa<AsinOp>(definingOp))
		return [operand = children[0], derOperand = derChildren[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			double operandValue = operand(tt, cj, yy, yp, ind, var);
			return derOperand(tt, cj, yy, yp, ind, var) /
						 (std::sqrt(1 - operandValue * operandValue));
		};

	if (mlir::isa<AcosOp>(definingOp))
		return [operand = children[0], derOperand = derChildren[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			double operandValue = operand(tt, cj, yy, yp, ind, var);
			return -derOperand(tt, cj, yy, yp, ind, var) /
						 (std::sqrt(1 - operandValue * operandValue));
		};

	if (mlir::isa<AtanOp>(definingOp))
		return [operand = children[0], derOperand = derChildren[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			double operandValue = operand(tt, cj, yy, yp, ind, var);
			return derOperand(tt, cj, yy, yp, ind, var) /
						 (1 + operandValue * operandValue);
		};

	if (mlir::isa<SinhOp>(definingOp))
		return [operand = children[0], derOperand = derChildren[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::cosh(operand(tt, cj, yy, yp, ind, var)) *
						 derOperand(tt, cj, yy, yp, ind, var);
		};

	if (mlir::isa<CoshOp>(definingOp))
		return [operand = children[0], derOperand = derChildren[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			return std::sinh(operand(tt, cj, yy, yp, ind, var)) *
						 derOperand(tt, cj, yy, yp, ind, var);
		};

	if (mlir::isa<TanhOp>(definingOp))
		return [operand = children[0], derOperand = derChildren[0]](
							 double tt,
							 double cj,
							 double *yy,
							 double *yp,
							 Indexes &ind,
							 double var) -> double {
			double tanhOperandValue = std::tanh(operand(tt, cj, yy, yp, ind, var));
			return (1 - tanhOperandValue * tanhOperandValue) *
						 derOperand(tt, cj, yy, yp, ind, var);
		};

	// TODO: Handle CallOp

	assert(false && "Unexpected operation");
}

int IdaSolver::residualFunction(
		realtype tt, N_Vector yy, N_Vector yp, N_Vector rr, void *user_data)
{
	realtype *yval = N_VGetArrayPointer(yy);
	realtype *ypval = N_VGetArrayPointer(yp);
	realtype *rval = N_VGetArrayPointer(rr);

	UserData *data = static_cast<UserData *>(user_data);

	// For every vector equation
	for (size_t eq = 0; eq < data->dimensions.size(); eq++)
	{
		bool finished = false;

		// Initialize the multidimensional interval of the vector equation
		Indexes indexes;
		for (size_t dim = 0; dim < data->dimensions[eq].size(); dim++)
		{
			indexes.push_back(data->dimensions[eq][dim].first);
		}

		// For every scalar equation in the vector equation
		while (!finished)
		{
			// Compute i-th residual function
			*rval++ = data->residuals[eq](tt, -1, yval, ypval, indexes, -1);

			// Update multidimensional interval, exit while loop if finished
			for (int dim = data->dimensions[eq].size() - 1; dim >= 0; dim--)
			{
				indexes[dim]++;
				if (indexes[dim] == data->dimensions[eq][dim].second)
				{
					if (dim == 0)
						finished = true;
					else
						indexes[dim] = data->dimensions[eq][dim].first;
				}
				else
				{
					break;
				}
			}
		}
	}

	return 0;
}

int IdaSolver::jacobianMatrix(
		realtype tt,
		realtype cj,
		N_Vector yy,
		N_Vector yp,
		N_Vector rr,
		SUNMatrix JJ,
		void *user_data,
		N_Vector tempv1,
		N_Vector tempv2,
		N_Vector tempv3)
{
	assert(SUNSparseMatrix_SparseType(JJ) == CSR_MAT);

	realtype *yval = N_VGetArrayPointer(yy);
	realtype *ypval = N_VGetArrayPointer(yp);

	sunindextype *rowptrs = SUNSparseMatrix_IndexPointers(JJ);
	sunindextype *colvals = SUNSparseMatrix_IndexValues(JJ);

	realtype *jacobian = SUNSparseMatrix_Data(JJ);
	// SUNMatZero(JJ);

	UserData *data = static_cast<UserData *>(user_data);

	size_t totalNnzElements = 0;
	*rowptrs++ = totalNnzElements;

	// For every vector equation
	for (size_t eq = 0; eq < data->dimensions.size(); eq++)
	{
		bool finished = false;

		// Initialize the multidimensional interval of the vector equation
		Indexes indexes;
		for (size_t dim = 0; dim < data->dimensions[eq].size(); dim++)
		{
			indexes.push_back(data->dimensions[eq][dim].first);
		}

		// For every scalar equation in the vector equation
		while (!finished)
		{
			totalNnzElements += data->rowLength[eq];
			*rowptrs++ = totalNnzElements;

			// For every variable with respect to which every equation must be
			// partially differentiated
			for (size_t var = 0; var < data->rowLength[eq]; var++)
			{
				// Compute i-th jacobian value
				*jacobian++ =
						data->jacobianMatrix[eq](tt, cj, yval, ypval, indexes, var);
				*colvals++ = var;
			}

			// Update multidimensional interval, exit while loop if finished
			for (int dim = data->dimensions[eq].size() - 1; dim >= 0; dim--)
			{
				indexes[dim]++;
				if (indexes[dim] == data->dimensions[eq][dim].second)
				{
					if (dim == 0)
						finished = true;
					else
						indexes[dim] = data->dimensions[eq][dim].first;
				}
				else
				{
					break;
				}
			}
		}
	}

	return 0;
}

mlir::LogicalResult IdaSolver::checkRetval(
		void *retval, const char *funcname, int opt)
{
	// Check if SUNDIALS function returned NULL pointer (no memory allocated)
	if (opt == 0 && retval == NULL)
	{
		llvm::errs() << "SUNDIALS_ERROR: " << funcname
								 << "() failed - returned NULL pointer\n";
		return mlir::LogicalResult::failure();
	}

	// Check if SUNDIALS function returned a positive integer value
	if (opt == 1 && *((int *) retval) < 0)
	{
		llvm::errs() << "SUNDIALS_ERROR: " << funcname
								 << "() failed  with return value = " << *(int *) retval
								 << "\n";
		return mlir::LogicalResult::failure();
	}

	// Check if function returned NULL pointer (no memory allocated)
	if (opt == 2 && retval == NULL)
	{
		llvm::errs() << "SUNDIALS_ERROR: " << funcname
								 << "() failed - returned NULL pointer\n";
		return mlir::LogicalResult::failure();
	}

	return mlir::LogicalResult::success();
}
