#include <llvm/ADT/SmallVector.h>
#include <marco/ida/IdaMangling.h>
#include <marco/ida/IdaSolver.h>
#include <marco/mlirlowerer/passes/model/BltBlock.h>
#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Expression.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/ReferenceMatcher.h>
#include <marco/mlirlowerer/passes/model/Variable.h>
#include <marco/mlirlowerer/passes/model/VectorAccess.h>
#include <marco/runtime/IdaFunctions.h>
#include <marco/utils/Interval.hpp>

using namespace marco::codegen::ida;
using namespace marco::codegen::model;
using namespace marco::codegen::modelica;

static realtype getValue(ConstantOp constantOp)
{
	mlir::Attribute attribute = constantOp.value();

	if (auto integer = attribute.dyn_cast<IntegerAttribute>())
		return integer.getValue();

	assert(attribute.isa<RealAttribute>());

	return attribute.cast<RealAttribute>().getValue();
}

static sunindextype computeNEQ(const Model& model)
{
	sunindextype result = 0;

	for (const BltBlock& bltBlock : model.getBltBlocks())
		result += bltBlock.equationsCount();

	return result;
}

IdaSolver::IdaSolver(
		const Model& model,
		realtype startTime,
		realtype endTime,
		realtype timeStep,
		realtype relativeTolerance,
		realtype absoluteTolerance)
		: model(model), endTime(endTime), forEquationsNumber(0)
{
	userData = allocIdaUserData(computeNEQ(model));
	addTime(userData, startTime, endTime, timeStep);
	addTolerance(userData, relativeTolerance, absoluteTolerance);
}

mlir::LogicalResult IdaSolver::init()
{
	sunindextype varOffset = 0;

	model.getOp().init().walk([&](FillOp fillOp) {
		// Map all vector variables to their initial value.
		if (!model.hasVariable(fillOp.memory()))
			return;

		Variable var = model.getVariable(fillOp.memory());

		if (var.isDerivative())
			return;

		mlir::Operation* op = fillOp.value().getDefiningOp();
		ConstantOp constantOp = mlir::cast<ConstantOp>(op);
		realtype value = getValue(constantOp);

		initialValueMap[var] = value;
		if (var.isState())
			initialValueMap[model.getVariable(var.getDer())] = value;
	});

	model.getOp().init().walk([&](AssignmentOp assignmentOp) {
		mlir::Operation* op = assignmentOp.destination().getDefiningOp();

		if (SubscriptionOp subscriptionOp = mlir::dyn_cast<SubscriptionOp>(op))
		{
			// Map all scalar variables to their initial value.
			if (!model.hasVariable(subscriptionOp.source()))
				return;

			Variable var = model.getVariable(subscriptionOp.source());

			if (var.isDerivative())
				return;

			op = assignmentOp.source().getDefiningOp();
			ConstantOp constantOp = mlir::cast<ConstantOp>(op);
			realtype value = getValue(constantOp);

			initialValueMap[var] = value;
			if (var.isState())
				initialValueMap[model.getVariable(var.getDer())] = value;
		}
		else if (AllocOp allocOp = mlir::dyn_cast<AllocOp>(op))
		{
			// Initialize all other vector variables to zero.
			if (!model.hasVariable(allocOp))
				return;

			Variable var = model.getVariable(allocOp);

			if (var.isDerivative())
				return;

			initialValueMap[var] = 0.0;
			if (var.isState())
				initialValueMap[model.getVariable(var.getDer())] = 0.0;
		}
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
				sunindextype varIndex = addVarOffset(userData, varOffset);
				variableIndexMap[var] = varIndex;

				if (var.isState())
					variableIndexMap[model.getVariable(var.getDer())] = varIndex;
				else if (var.isDerivative())
					variableIndexMap[model.getVariable(var.getState())] = varIndex;

				// Add dimensions of the variable to the ida user data.
				marco::MultiDimInterval multiDimInterval = var.toMultiDimInterval();
				llvm::SmallVector<sunindextype, 3> dimensions;

				for (Interval& interval : multiDimInterval)
					dimensions.push_back(interval.size());

				ArrayDescriptor<sunindextype, 1> dims(
						&dimensions[0], { dimensions.size() });
				addVarDimension(userData, dims);

				// Initialize variablesValues, derivativesValues, idValues.
				setInitialValue(
						userData,
						varIndex,
						var.toMultiDimInterval().size(),
						initialValueMap[var],
						var.isState() || var.isDerivative());

				// Increase the length of the current row.
				varOffset += var.toMultiDimInterval().size();
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

				assert(variableIndexMap.find(var) != variableIndexMap.end());
				VectorAccess vectorAccess =
						AccessToVar::fromExp(path.getExpression()).getAccess();

				// If the variable access has not been insterted yet, initialize it.
				if (accessesMap.find({ var, vectorAccess }) == accessesMap.end())
				{
					// Compute the access offset based on the induction variables of the
					// for-equation.
					llvm::SmallVector<sunindextype, 3> offsets;
					llvm::SmallVector<sunindextype, 3> inductions;

					for (auto& acc : vectorAccess.getMappingOffset())
					{
						sunindextype accOffset =
								acc.isDirectAccess() ? acc.getOffset() : (acc.getOffset() + 1);
						sunindextype accInduction =
								acc.isOffset() ? acc.getInductionVar() : -1;
						offsets.push_back(accOffset);
						inductions.push_back(accInduction);
					}

					ArrayDescriptor<sunindextype, 1> unsizedAcc(
							&offsets[0], { offsets.size() });
					ArrayDescriptor<sunindextype, 1> unsizedInd(
							&inductions[0], { inductions.size() });

					// Add accesses of the variable to the ida user data.
					sunindextype accessIndex = addVarAccess(
							userData, variableIndexMap[var], unsizedAcc, unsizedInd);

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

	assert(varOffset == getEquationsNumber());

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

		if (getIdaTime(userData) >= endTime - 1e-12)
			return mlir::success();
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

	for (sunindextype i : irange(getEquationsNumber()))
		OS << ", " << getVariable(i);

	OS << "\n";
}

void IdaSolver::printStats(llvm::raw_ostream& OS)
{
	sunindextype nst = numSteps(userData);
	sunindextype nre = numResEvals(userData);
	sunindextype nje = numJacEvals(userData);
	sunindextype nni = numNonlinIters(userData);

	OS << "\nFinal Run Statistics:\n";
	OS << "Number of for-equations        = " << getForEquationsNumber() << "\n";
	OS << "Number of scalar equations     = " << getEquationsNumber() << "\n";
	OS << "Number of non-zero values      = " << getNonZeroValuesNumber() << "\n";
	OS << "Number of steps                = " << nst << "\n";
	OS << "Number of residual evaluations = " << nre << "\n";
	OS << "Number of Jacobian evaluations = " << nje << "\n";
	OS << "Number of nonlinear iterations = " << nni << "\n";
}

void IdaSolver::printIncidenceMatrix(llvm::raw_ostream& OS)
{
	OS << getIncidenceMatrix(userData);
}

sunindextype IdaSolver::getForEquationsNumber() { return forEquationsNumber; }

sunindextype IdaSolver::getEquationsNumber()
{
	return getNumberOfEquations(userData);
}

sunindextype IdaSolver::getNonZeroValuesNumber()
{
	return getNumberOfNonZeroValues(userData);
}

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

void IdaSolver::getDimension(const Equation& equation)
{
	llvm::SmallVector<sunindextype, 3> start;
	llvm::SmallVector<sunindextype, 3> end;

	for (marco::Interval& interval : equation.getInductions())
	{
		start.push_back(interval.min() - 1);
		end.push_back(interval.max() - 1);
	}

	ArrayDescriptor<sunindextype, 1> unsizedMins(&start[0], { start.size() });
	ArrayDescriptor<sunindextype, 1> unsizedMaxes(&end[0], { end.size() });

	addEqDimension(userData, unsizedMins, unsizedMaxes);
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
	llvm::SmallVector<sunindextype, 2> children;
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

	assert(false && "CallOp is not supported in this tool");
}
