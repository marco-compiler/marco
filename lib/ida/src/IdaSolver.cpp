#include <marco/ida/IdaSolver.h>
#include <marco/mlirlowerer/passes/model/BltBlock.h>
#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Expression.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/ReferenceMatcher.h>
#include <marco/mlirlowerer/passes/model/Variable.h>
#include <marco/mlirlowerer/passes/model/VectorAccess.h>
#include <marco/runtime/Runtime.h>
#include <marco/utils/Interval.hpp>

using namespace marco::codegen::ida;
using namespace marco::codegen::model;
using namespace marco::codegen::modelica;

static realtype getValue(ConstantOp constantOp)
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
		const Model& model,
		realtype startTime,
		realtype stopTime,
		realtype relativeTolerance,
		realtype absoluteTolerance)
		: model(model),
			stopTime(stopTime),
			forEquationsNumber(0),
			equationsNumber(computeNEQ()),
			nonZeroValuesNumber(computeNNZ())
{
	userData = allocIdaUserData(equationsNumber, nonZeroValuesNumber);
	addTime(userData, startTime, stopTime);
	addTolerance(userData, relativeTolerance, absoluteTolerance);
}

mlir::LogicalResult IdaSolver::init()
{
	sunindextype varOffset = 0;

	// TODO: Add different value handling for initialization

	// Map all vector variables to their initial value.
	model.getOp().init().walk([&](FillOp fillOp) {
		if (!model.hasVariable(fillOp.memory()))
			return;

		Variable var = model.getVariable(fillOp.memory());

		if (var.isDerivative())
			return;

		mlir::Operation* op = fillOp.value().getDefiningOp();
		ConstantOp constantOp = mlir::dyn_cast<ConstantOp>(op);
		realtype value = getValue(constantOp);

		initialValueMap[var] = value;
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

		if (var.isDerivative())
			return;

		op = assignmentOp.source().getDefiningOp();
		ConstantOp constantOp = mlir::dyn_cast<ConstantOp>(op);
		realtype value = getValue(constantOp);

		initialValueMap[var] = value;
		if (var.isState())
			initialValueMap[model.getVariable(var.getDer())] = value;
	});

	// Compute all non-trivial variable offsets and dimensions.
	for (const BltBlock& bltBlock : model.getBltBlocks())
	{
		for (const Equation& equation : bltBlock.getEquations())
		{
			// Get the variable matched with every equation.
			Variable var =
					model.getVariable(equation.getDeterminedVariable().getVar());

			assert(!var.isTrivial());

			// If the variable has not been insterted yet, initialize it.
			if (variableIndexMap.find(var) == variableIndexMap.end())
			{
				// Note the variable offset from the beginning of the variable array.
				sunindextype varIndex = addVariableOffset(userData, varOffset);
				variableIndexMap[var] = varIndex;

				if (var.isState())
					variableIndexMap[model.getVariable(var.getDer())] = varIndex;
				else if (var.isDerivative())
					variableIndexMap[model.getVariable(var.getState())] = varIndex;

				// Initialize variablesValues, derivativesValues, idValues.
				setInitialValue(
						userData,
						varIndex,
						var.toMultiDimInterval().size(),
						initialValueMap[var],
						var.isState() || var.isDerivative());

				// Increase the length of the current row.
				varOffset += var.toMultiDimInterval().size();

				// Compute the multi-dimensional offset of the array.
				marco::MultiDimInterval dimensions = var.toMultiDimInterval();
				std::vector<sunindextype> dims;
				for (size_t i = 1; i < dimensions.dimensions(); i++)
				{
					for (size_t j = 0; j < dims.size(); j++)
						dims[j] *= dimensions.at(i).size();
					dims.push_back(dimensions.at(i).size());
				}
				dims.push_back(1);

				// Add dimensions of the variable to the ida user data.
				for (size_t i = 0; i < dims.size(); i++)
					addVariableDimension(userData, varIndex, dims[i]);
			}
		}
	}

	// Compute all non-trivial variable accesses of each equation.
	for (const BltBlock& bltBlock : model.getBltBlocks())
	{
		for (const Equation& equation : bltBlock.getEquations())
		{
			ReferenceMatcher matcher(equation);
			std::set<std::pair<Variable, VectorAccess>> varSet;

			// Add all different variable accesses to a set.
			for (ExpressionPath& path : matcher)
			{
				Variable var =
						model.getVariable(path.getExpression().getReferredVectorAccess());
				if (var.isTime())
					continue;

				VectorAccess acc =
						AccessToVar::fromExp(path.getExpression()).getAccess();

				if (var.isDerivative())
					varSet.insert({ model.getVariable(var.getState()), acc });
				else
					varSet.insert({ var, acc });
			}

			// Add to IDA the number of non-zero values of the current equation.
			sunindextype rowIndex = addRowLength(userData, varSet.size());

			for (ExpressionPath& path : matcher)
			{
				Variable var =
						model.getVariable(path.getExpression().getReferredVectorAccess());

				if (var.isTime())
					continue;

				VectorAccess vectorAccess =
						AccessToVar::fromExp(path.getExpression()).getAccess();

				// If the variable access has not been insterted yet, initialize it.
				if (accessesMap.find({ var, vectorAccess }) == accessesMap.end())
				{
					// Compute the access offset based on the induction variables of the
					// for-equation.
					std::vector<std::pair<sunindextype, sunindextype>> access;

					for (auto& acc : vectorAccess.getMappingOffset())
					{
						sunindextype accOffset =
								acc.isDirectAccess() ? acc.getOffset() : acc.getOffset() + 1;
						sunindextype accInduction =
								acc.isOffset() ? acc.getInductionVar() : -1;
						access.push_back({ accOffset, accInduction });
					}

					// Add accesses of the variable to the ida user data.
					sunindextype accessIndex = addNewVariableAccess(
							userData,
							variableIndexMap[var],
							access[0].first,
							access[0].second);
					for (size_t i = 1; i < access.size(); i++)
						addVariableAccess(
								userData, accessIndex, access[i].first, access[i].second);

					accessesMap[{ var, vectorAccess }] = accessIndex;

					if (var.isState())
						accessesMap[{ model.getVariable(var.getDer()), vectorAccess }] =
								accessIndex;
					else if (var.isDerivative())
						accessesMap[{ model.getVariable(var.getState()), vectorAccess }] =
								accessIndex;
				}

				// Add to IDA the indexes of non-zero values of the current equation.
				addColumnIndex(userData, rowIndex, accessesMap[{ var, vectorAccess }]);
			}
		}
	}

	// Add to IDA the dimensions of each equation and the lambda functions needed
	// to compute the residual function and the jacobian matrix of the system.
	for (const BltBlock& bltBlock : model.getBltBlocks())
	{
		for (const Equation& equation : bltBlock.getEquations())
		{
			getDimension(equation);
			getResidualAndJacobian(equation);
			forEquationsNumber++;
		}
	}

	assert(varOffset == equationsNumber);

	initialValueMap.clear();
	variableIndexMap.clear();
	accessesMap.clear();

	bool success = idaInit(userData);
	if (!success)
		return mlir::failure();
	return mlir::success();
}

bool IdaSolver::step() { return idaStep(userData); }

mlir::LogicalResult IdaSolver::run(llvm::raw_ostream& OS)
{
	while (true)
	{
		bool success = idaStep(userData);

		if (!success)
			return mlir::failure();

		printOutput(OS);

		if (getIdaTime(userData) >= stopTime)
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

	for (sunindextype i : irange(equationsNumber))
		OS << ", " << getVariable(i);

	OS << "\n";
}

void IdaSolver::printStats(llvm::raw_ostream& OS)
{
	sunindextype nst = numSteps(userData);
	sunindextype nre = numResEvals(userData);
	sunindextype nje = numJacEvals(userData);
	sunindextype nni = numNonlinIters(userData);

	OS << "\nFinal Run Statistics:\n\n";
	OS << "Number of for-equations            = " << forEquationsNumber << "\n";
	OS << "Number of scalar equations         = " << equationsNumber << "\n";
	OS << "Number of non-zero values          = " << nonZeroValuesNumber << "\n";
	OS << "Number of steps                    = " << nst << "\n";
	OS << "Number of residual evaluations     = " << nre << "\n";
	OS << "Number of Jacobian evaluations     = " << nje << "\n";
	OS << "Number of nonlinear iterations     = " << nni << "\n";
}

sunindextype IdaSolver::getForEquationsNumber() { return forEquationsNumber; }

sunindextype IdaSolver::getEquationsNumber() { return equationsNumber; }

sunindextype IdaSolver::getNonZeroValuesNumber() { return nonZeroValuesNumber; }

realtype IdaSolver::getTime() { return getIdaTime(userData); }

realtype IdaSolver::getVariable(sunindextype index)
{
	return getIdaVariable(userData, index);
}

realtype IdaSolver::getDerivative(sunindextype index)
{
	return getIdaDerivative(userData, index);
}

sunindextype IdaSolver::getRowLength(sunindextype index)
{
	return getIdaRowLength(userData, index);
}

IdaSolver::Dimension IdaSolver::getDimension(sunindextype index)
{
	return getIdaDimension(userData, index);
}

sunindextype IdaSolver::computeNEQ()
{
	sunindextype result = 0;

	for (const BltBlock& bltBlock : model.getBltBlocks())
		result += bltBlock.equationsCount();

	return result;
}

sunindextype IdaSolver::computeNNZ()
{
	sunindextype result = 0;

	// For each equation, compute how many different variables are accessed.
	for (const BltBlock& bltBlock : model.getBltBlocks())
	{
		for (const Equation& equation : bltBlock.getEquations())
		{
			ReferenceMatcher matcher(equation);
			std::set<std::pair<Variable, VectorAccess>> varSet;

			for (ExpressionPath& path : matcher)
			{
				Variable var =
						model.getVariable(path.getExpression().getReferredVectorAccess());
				if (var.isTime())
					continue;

				VectorAccess acc =
						AccessToVar::fromExp(path.getExpression()).getAccess();

				if (var.isDerivative())
					varSet.insert({ model.getVariable(var.getState()), acc });
				else
					varSet.insert({ var, acc });
			}

			result += varSet.size() * equation.getInductions().size();
		}
	}

	return result;
}

void IdaSolver::getDimension(const Equation& equation)
{
	for (marco::Interval& interval : equation.getInductions())
		addEquationDimension(
				userData, forEquationsNumber, interval.min() - 1, interval.max() - 1);
}

void IdaSolver::getResidualAndJacobian(const Equation& equation)
{
	sunindextype left = getFunction(equation.lhs());
	sunindextype right = getFunction(equation.rhs());

	addResidual(userData, left, right);
	addJacobian(userData, left, right);
}

sunindextype IdaSolver::getFunction(const Expression& expression)
{
	mlir::Operation* definingOp = expression.getOp();

	// Constant value.
	if (auto op = mlir::dyn_cast<ConstantOp>(definingOp))
	{
		realtype value = getValue(op);
		return lambdaConstant(userData, value);
	}

	// Induction argument.
	if (expression.isInduction())
	{
		return lambdaInduction(
				userData, expression.get<Induction>().getArgument().getArgNumber());
	}

	// Variable reference.
	if (expression.isReferenceAccess())
	{
		// Compute the IDA offset of the variable in the 1D array variablesVector.
		Variable var = model.getVariable(expression.getReferredVectorAccess());

		// Time variable
		if (var.isTime())
			return lambdaTime(userData);

		VectorAccess vectorAccess = AccessToVar::fromExp(expression).getAccess();

		assert(variableIndexMap.find(var) != variableIndexMap.end());
		assert(accessesMap.find({ var, vectorAccess }) != accessesMap.end());

		sunindextype accessIndex = accessesMap[{ var, vectorAccess }];

		if (var.isDerivative())
			return lambdaDerivative(userData, accessIndex);
		else
			return lambdaVariable(userData, accessIndex);
	}

	// Get the lambda functions to compute the values of all the children.
	std::vector<sunindextype> children;
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

	if (mlir::isa<Atan2Op>(definingOp))
		return lambdaAtan2(userData, children[0], children[1]);

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
