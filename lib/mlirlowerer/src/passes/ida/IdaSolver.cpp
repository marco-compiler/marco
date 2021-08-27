#include <marco/mlirlowerer/passes/ida/IdaSolver.h>
#include <marco/mlirlowerer/passes/model/BltBlock.h>
#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Expression.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/Variable.h>
#include <marco/mlirlowerer/passes/model/VectorAccess.h>
#include <marco/runtime/Runtime.h>
#include <marco/utils/Interval.hpp>

using namespace marco::codegen::ida;
using namespace marco::codegen::model;
using namespace marco::codegen::modelica;

static double getValue(ConstantOp constantOp)
{
	mlir::Attribute attribute = constantOp.value();

	if (auto integer = attribute.dyn_cast<IntegerAttribute>())
		return integer.getValue();

	if (auto real = attribute.dyn_cast<RealAttribute>())
		return real.getValue();

	assert(false && "Unreachable");
	return 0.0;
}

IdaSolver::IdaSolver(
		Model& model,
		double startTime,
		double stopTime,
		double relativeTolerance,
		double absoluteTolerance)
		: model(model), problemSize(0), equationsNumber(computeNEQ())
{
	userData = allocIdaUserData(equationsNumber, computeNNZ());
	addTime(userData, startTime, stopTime);
	addTolerances(userData, relativeTolerance, absoluteTolerance);
}

mlir::LogicalResult IdaSolver::init()
{
	int64_t rowLength = 0;

	// TODO: Add different value handling for initialization

	// Map all vector variables to their initial value.
	model.getOp().init().walk([&](FillOp fillOp) {
		if (!model.hasVariable(fillOp.memory()))
			return;

		Variable var = model.getVariable(fillOp.memory());

		mlir::Operation* op = fillOp.value().getDefiningOp();
		ConstantOp constantOp = mlir::dyn_cast<ConstantOp>(op);
		double value = getValue(constantOp);

		initialValueMap[var] = value;
		assert(!var.isDerivative());
		if (var.isState())
			initialValueMap[model.getVariable(var.getDer())] = value;
	});

	// Map all scalar variables to their initial value.
	model.getOp().init().walk([&](AssignmentOp assignmentOp) {
		mlir::Operation* op = assignmentOp.destination().getDefiningOp();
		SubscriptionOp subscriptionOp = mlir::dyn_cast<SubscriptionOp>(op);

		if (!model.hasVariable(subscriptionOp.source()))
			return;

		Variable var = model.getVariable(subscriptionOp.source());

		op = assignmentOp.source().getDefiningOp();
		ConstantOp constantOp = mlir::dyn_cast<ConstantOp>(op);
		double value = getValue(constantOp);

		initialValueMap[var] = value;
		assert(!var.isDerivative());
		if (var.isState())
			initialValueMap[model.getVariable(var.getDer())] = value;
	});

	for (BltBlock& bltBlock : model.getBltBlocks())
	{
		for (Equation& equation : bltBlock.getEquations())
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
				for (int64_t i : irange(var.toMultiDimInterval().size()))
				{
					setInitialValues(
							userData,
							rowLength + i,
							initialValueMap[var],
							var.isState() || var.isDerivative());
				}

				// Increase the length of the current row.
				rowLength += var.toMultiDimInterval().size();
			}
		}

		// Initialize UserData with all parameters needed by IDA.
		for (Equation& equation : bltBlock.getEquations())
		{
			addRowLength(userData, rowLength);
			getDimension(equation);
			getResidualAndJacobian(equation);
			problemSize++;
		}
	}

	assert(rowLength == equationsNumber);

	initialValueMap.clear();
	indexOffsetMap.clear();

	bool success = idaInit(userData);
	if (!success)
		return mlir::failure();
	return mlir::success();
}

int64_t IdaSolver::step() { return idaStep(userData); }

mlir::LogicalResult IdaSolver::run(llvm::raw_ostream& OS)
{
	while (true)
	{
		int64_t isFinished = idaStep(userData);

		if (isFinished == -1)
			return mlir::failure();

		printOutput(OS);

		if (isFinished == 0)
		{
			printStats(OS);
			return mlir::success();
		}
	}
}

mlir::LogicalResult IdaSolver::free()
{
	bool success = freeIdaUserData(userData);
	if (!success)
		return mlir::failure();
	return mlir::success();
}

void IdaSolver::printOutput(llvm::raw_ostream& OS)
{
	OS << getTime();

	for (int64_t i : irange(equationsNumber))
		OS << ", " << getVariable(i);

	OS << "\n";
}

void IdaSolver::printStats(llvm::raw_ostream& OS)
{
	int64_t nst = numSteps(userData);
	int64_t nre = numResEvals(userData);
	int64_t nje = numJacEvals(userData);
	int64_t nni = numNonlinIters(userData);

	OS << "\nFinal Run Statistics:\n\n";
	OS << "Number of steps                    = " << nst << "\n";
	OS << "Number of residual evaluations     = " << nre << "\n";
	OS << "Number of Jacobian evaluations     = " << nje << "\n";
	OS << "Number of nonlinear iterations     = " << nni << "\n";
}

int64_t IdaSolver::getProblemSize() { return problemSize; }

int64_t IdaSolver::getEquationsNumber() { return equationsNumber; }

double IdaSolver::getTime() { return getIdaTime(userData); }

double IdaSolver::getVariable(int64_t index)
{
	return getIdaVariable(userData, index);
}

double IdaSolver::getDerivative(int64_t index)
{
	return getIdaDerivative(userData, index);
}

int64_t IdaSolver::getRowLength(int64_t index)
{
	return getIdaRowLength(userData, index);
}

std::vector<std::pair<int64_t, int64_t>> IdaSolver::getDimension(int64_t index)
{
	return getIdaDimension(userData, index);
}

int64_t IdaSolver::computeNEQ()
{
	int64_t result = 0;

	for (BltBlock& bltBlock : model.getBltBlocks())
		result += bltBlock.size();

	return result;
}

int64_t IdaSolver::computeNNZ()
{
	int64_t result = 0, rowLength = 0;
	std::set<Variable> varSet;

	for (BltBlock& bltBlock : model.getBltBlocks())
	{
		for (Equation& equation : bltBlock.getEquations())
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

void IdaSolver::getDimension(const Equation& equation)
{
	for (marco::Interval& interval : equation.getInductions())
		addDimension(userData, problemSize, interval.min(), interval.max());
}

void IdaSolver::getResidualAndJacobian(const Equation& equation)
{
	int64_t left = getFunction(equation.lhs());
	int64_t right = getFunction(equation.rhs());

	addResidual(userData, left, right);
	addJacobian(userData, left, right);
}

int64_t IdaSolver::getFunction(const Expression& expression)
{
	mlir::Operation* definingOp = expression.getOp();

	// Constant value.
	if (auto op = mlir::dyn_cast<ConstantOp>(definingOp))
	{
		double value = getValue(op);
		return lambdaConstant(userData, value);
	}

	// Scalar variable reference.
	if (expression.isReference())
	{
		// Time variable
		Variable var = model.getVariable(expression.getReferredVectorAccess());
		if (indexOffsetMap.find(var) == indexOffsetMap.end())
			return lambdaTime(userData);

		int64_t offset = indexOffsetMap[var];

		if (var.isDerivative())
			return lambdaScalarDerivative(userData, offset);
		else
			return lambdaScalarVariable(userData, offset);
	}

	assert(expression.isOperation());

	// Vector variable reference.
	if (expression.isReferenceAccess())
	{
		// Compute the IDA offset of the variable in the 1D array variablesVector.
		Variable var = model.getVariable(expression.getReferredVectorAccess());
		assert(indexOffsetMap.find(var) != indexOffsetMap.end());
		int64_t offset = indexOffsetMap[var];

		// Compute the access offset based on the induction variables of the
		// for-equation.
		VectorAccess vectorAccess = AccessToVar::fromExp(expression).getAccess();
		std::vector<std::pair<int64_t, int64_t>> access;

		for (auto& acc : vectorAccess.getMappingOffset())
		{
			int64_t accOffset =
					acc.isDirectAccess() ? acc.getOffset() : acc.getOffset() + 1;
			int64_t accInduction = acc.isOffset() ? acc.getInductionVar() : -1;
			access.push_back({ accOffset, accInduction });
		}

		// Compute the multi-dimensional offset of the array.
		marco::MultiDimInterval dimensions = var.toMultiDimInterval();
		std::vector<int64_t> dim;
		for (size_t i = 1; i < dimensions.dimensions(); i++)
		{
			for (size_t j = 0; j < dim.size(); j++)
				dim[j] *= dimensions.at(i).size();
			dim.push_back(dimensions.at(i).size());
		}
		dim.push_back(1);

		// Add accesses and dimensions of the variable to the ida user data.
		int64_t index =
				addNewLambdaAccess(userData, access[0].first, access[0].second);
		for (size_t i = 1; i < access.size(); i++)
			addLambdaAccess(userData, index, access[i].first, access[i].second);
		for (size_t i = 0; i < dim.size(); i++)
			addLambdaDimension(userData, index, dim[i]);

		assert(access.size() == dim.size());

		if (var.isDerivative())
			return lambdaVectorDerivative(userData, offset, index);
		else
			return lambdaVectorVariable(userData, offset, index);
	}

	// Get the lambda functions to compute the values of all the children.
	std::vector<int64_t> children;
	for (size_t i : marco::irange(expression.childrenCount()))
		children.push_back(getFunction(expression.getChild(i)));

	if (mlir::isa<NegateOp>(definingOp))
		return lambdaNegate(userData, children[0]);

	if (mlir::isa<AddOp>(definingOp))
		return lambdaAdd(userData, children[0], children[1]);

	if (mlir::isa<SubOp>(definingOp))
		return lambdaSub(userData, children[0], children[1]);

	if (mlir::isa<MulOp>(definingOp))
		return lambdaMul(userData, children[0], children[1]);

	if (mlir::isa<DivOp>(definingOp))
		return lambdaDiv(userData, children[0], children[1]);

	if (mlir::isa<PowOp>(definingOp))
		return lambdaPow(userData, children[0], children[1]);

	if (mlir::isa<AbsOp>(definingOp))
		return lambdaAbs(userData, children[0]);

	if (mlir::isa<SignOp>(definingOp))
		return lambdaSign(userData, children[0]);

	if (mlir::isa<SqrtOp>(definingOp))
		return lambdaSqrt(userData, children[0]);

	if (mlir::isa<ExpOp>(definingOp))
		return lambdaExp(userData, children[0]);

	if (mlir::isa<LogOp>(definingOp))
		return lambdaLog(userData, children[0]);

	if (mlir::isa<Log10Op>(definingOp))
		return lambdaLog10(userData, children[0]);

	if (mlir::isa<SinOp>(definingOp))
		return lambdaSin(userData, children[0]);

	if (mlir::isa<CosOp>(definingOp))
		return lambdaCos(userData, children[0]);

	if (mlir::isa<TanOp>(definingOp))
		return lambdaTan(userData, children[0]);

	if (mlir::isa<AsinOp>(definingOp))
		return lambdaAsin(userData, children[0]);

	if (mlir::isa<AcosOp>(definingOp))
		return lambdaAcos(userData, children[0]);

	if (mlir::isa<AtanOp>(definingOp))
		return lambdaAtan(userData, children[0]);

	if (mlir::isa<SinhOp>(definingOp))
		return lambdaSinh(userData, children[0]);

	if (mlir::isa<CoshOp>(definingOp))
		return lambdaCosh(userData, children[0]);

	if (mlir::isa<TanhOp>(definingOp))
		return lambdaTanh(userData, children[0]);

	// TODO: Handle CallOp

	assert(false && "Unexpected operation");
}
