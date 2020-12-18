#include <gtest/gtest.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/mlirlowerer/Runner.hpp>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
using namespace std;

static mlir::ModuleOp wrapFunctionWithModule(mlir::MLIRContext& context, mlir::FuncOp& function)
{
	mlir::OpBuilder builder(&context);
	mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
	module.push_back(function);
	return module;
}

static mlir::FuncOp getFunctionReturningValue(mlir::MLIRContext& context, Type& returnType, function<mlir::Value(MlirLowerer&)> callback)
{
	MlirLowerer lowerer(context);
	auto& builder = lowerer.getOpBuilder();
	auto functionType = builder.getFunctionType({}, lowerer.lower(returnType));
	mlir::FuncOp function = mlir::FuncOp::create(builder.getUnknownLoc(), "main", functionType);
	auto &entryBlock = *function.addEntryBlock();
	builder.setInsertionPointToStart(&entryBlock);
	builder.create<mlir::ReturnOp>(builder.getUnknownLoc(), callback(lowerer));
	return function;
}

TEST(MathOps, addSameSignIntegers)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::add>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(2)),
			Expression(location, Type::Int(), Constant(3)));

	mlir::MLIRContext context;

	mlir::FuncOp function = getFunctionReturningValue(
			context,
			expression.getType(),
			[&](MlirLowerer& lowerer) -> mlir::Value
			{
				auto values = lowerer.lower<modelica::Expression>(expression);
				EXPECT_EQ(values.size(), 1);
				return *values[0];
			});

	Runner runner(&context, wrapFunctionWithModule(context, function));
	int result = 0;
	runner.run("main", result);
	EXPECT_EQ(result, 5);
}

TEST(MathOps, addDifferentSignIntegers)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::add>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(2)),
			Expression(location, Type::Int(), Constant(-3)));

	mlir::MLIRContext context;

	mlir::FuncOp function = getFunctionReturningValue(
			context,
			expression.getType(),
			[&](MlirLowerer& lowerer) -> mlir::Value
			{
				auto values = lowerer.lower<modelica::Expression>(expression);
				EXPECT_EQ(values.size(), 1);
				return *values[0];
			});

	Runner runner(&context, wrapFunctionWithModule(context, function));
	int result = 0;
	runner.run("main", result);
	EXPECT_EQ(result, -1);
}

TEST(MathOps, addSameSignFloats)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::add>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(2.0)),
			Expression(location, Type::Float(), Constant(3.0)));

	mlir::MLIRContext context;

	mlir::FuncOp function = getFunctionReturningValue(
			context,
			expression.getType(),
			[&](MlirLowerer& lowerer) -> mlir::Value
			{
				auto values = lowerer.lower<modelica::Expression>(expression);
				EXPECT_EQ(values.size(), 1);
				return *values[0];
			});

	Runner runner(&context, wrapFunctionWithModule(context, function));
	float result = 0;
	runner.run("main", result);
	EXPECT_EQ(result, 5.0);
}

TEST(MathOps, addDifferentSignFloats)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::add>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(2.0)),
			Expression(location, Type::Float(), Constant(-3.0)));

	mlir::MLIRContext context;

	mlir::FuncOp function = getFunctionReturningValue(
			context,
			expression.getType(),
			[&](MlirLowerer& lowerer) -> mlir::Value
			{
				auto values = lowerer.lower<modelica::Expression>(expression);
				EXPECT_EQ(values.size(), 1);
				return *values[0];
			});

	Runner runner(&context, wrapFunctionWithModule(context, function));
	float result = 0;
	runner.run("main", result);
	EXPECT_EQ(result, -1.0);
}

TEST(MathOps, addIntegerCastedToFloat)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::add>(
			location,
			Type::Float(),
			Expression(location, Type::Int(), Constant(2)),
			Expression(location, Type::Float(), Constant(-3.0)));

	mlir::MLIRContext context;

	mlir::FuncOp function = getFunctionReturningValue(
			context,
			expression.getType(),
			[&](MlirLowerer& lowerer) -> mlir::Value
			{
				auto values = lowerer.lower<modelica::Expression>(expression);
				EXPECT_EQ(values.size(), 1);
				return *values[0];
			});

	Runner runner(&context, wrapFunctionWithModule(context, function));
	float result = 0;
	runner.run("main", result);
	EXPECT_EQ(result, -1.0);
}
