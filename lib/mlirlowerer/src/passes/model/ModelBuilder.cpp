#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/SolveModel.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/ModelBuilder.h>
#include <modelica/utils/Interval.hpp>
#include <variant>

using namespace modelica::codegen::model;

ModelBuilder::ModelBuilder(Model& model) : model(model)
{
}

void ModelBuilder::lower(EquationOp equation)
{
	auto& body = equation.body();
	auto terminator = mlir::cast<EquationSidesOp>(body.front().getTerminator());

	// Left-hand side of the equation
	mlir::ValueRange lhs = terminator.lhs();
	assert(lhs.size() == 1);

	llvm::SmallVector<Expression, 3> lhsExpr;

	for (auto value : lhs)
		lhsExpr.push_back(lower(value));

	// Right-hand side of the equation
	mlir::ValueRange rhs = terminator.rhs();
	assert(rhs.size() == 1);

	llvm::SmallVector<Expression, 3> rhsExpr;

	for (auto value : rhs)
		rhsExpr.push_back(lower(value));

	// Number of values of left-hand side and right-hand side must match
	assert(lhsExpr.size() == rhsExpr.size());

	model.addEquation(Equation(
			equation,
			lhsExpr[0], rhsExpr[0],
			"eq_" + std::to_string(model.getEquations().size())));
}

void ModelBuilder::lower(ForEquationOp forEquation)
{
	auto& body = forEquation.body();
	auto terminator = mlir::cast<EquationSidesOp>(body.front().getTerminator());

	// Left-hand side of the equation
	mlir::ValueRange lhs = terminator.lhs();
	assert(lhs.size() == 1);

	llvm::SmallVector<Expression, 3> lhsExpr;

	for (auto value : lhs)
		lhsExpr.push_back(lower(value));

	// Right-hand side of the equation
	mlir::ValueRange rhs = terminator.rhs();
	assert(rhs.size() == 1);

	llvm::SmallVector<Expression, 3> rhsExpr;

	for (auto value : rhs)
		rhsExpr.push_back(lower(value));

	// Number of values of left-hand side and right-hand side must match
	assert(lhsExpr.size() == rhsExpr.size());

	llvm::SmallVector<Interval> intervals;

	for (auto induction : forEquation.inductions())
	{
		auto inductionOp = induction.getDefiningOp<InductionOp>();
		intervals.emplace_back(inductionOp.start(), inductionOp.end() + 1);
	}

	model.addEquation(Equation(
			forEquation,
			lhsExpr[0], rhsExpr[0],
			"eq_" + std::to_string(model.getEquations().size()),
			MultiDimInterval(intervals)));
}

Expression ModelBuilder::lower(mlir::Value value)
{
	mlir::Operation* definingOp = value.getDefiningOp();

	if (auto op = mlir::dyn_cast<LoadOp>(definingOp))
		return lower(op.memory());

	if (auto op = mlir::dyn_cast<AllocaOp>(definingOp))
	{
		model.addVariable(value);
		return Expression::reference(op);
	}

	if (auto op = mlir::dyn_cast<AllocOp>(definingOp))
	{
		model.addVariable(value);
		return Expression::reference(op);
	}

	if (auto op = mlir::dyn_cast<ConstantOp>(definingOp))
		return Expression::constant(op);

	llvm::SmallVector<Expression, 3> args;

	if (auto op = mlir::dyn_cast<CallOp>(definingOp))
	{
		for (auto arg : op.args())
			args.push_back(lower(arg));

		return Expression::operation(op, args);
	}

	if (auto op = mlir::dyn_cast<SubscriptionOp>(definingOp))
		return Expression::operation(op, lower(op.source()));

	if (auto op = mlir::dyn_cast<DerOp>(definingOp))
		return Expression::operation(op, lower(op.operand()));

	if (auto op = mlir::dyn_cast<NegateOp>(definingOp))
		return Expression::operation(op, lower(op.operand()));

	if (auto op = mlir::dyn_cast<AddOp>(definingOp))
		return Expression::operation(op, lower(op.lhs()), lower(op.rhs()));

	if (auto op = mlir::dyn_cast<SubOp>(definingOp))
		return Expression::operation(op, lower(op.lhs()), lower(op.rhs()));

	if (auto op = mlir::dyn_cast<MulOp>(definingOp))
		return Expression::operation(op, lower(op.lhs()), lower(op.rhs()));

	if (auto op = mlir::dyn_cast<DivOp>(definingOp))
		return Expression::operation(op, lower(op.lhs()), lower(op.rhs()));

	assert(false && "Unexpected operation");
}
