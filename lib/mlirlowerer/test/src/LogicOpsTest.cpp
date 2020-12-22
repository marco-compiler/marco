#include <gtest/gtest.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/mlirlowerer/Runner.hpp>
#include <modelica/utils/SourceRange.hpp>

#include "TestUtils.hpp"

using namespace modelica;
using namespace std;

TEST(LogicOps, andBooleansTrueOperands)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::land>(
			location,
			Type::Bool(),
			Expression::trueExp(location),
			Expression::trueExp(location));

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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(LogicOps, andBooleansTrueAndFalseOperands)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::land>(
			location,
			Type::Bool(),
			Expression::trueExp(location),
			Expression::falseExp(location));

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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(LogicOps, andBooleansFalseOperands)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::land>(
			location,
			Type::Bool(),
			Expression::falseExp(location),
			Expression::falseExp(location));

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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(LogicOps, andIntegers)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::land>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(13)),
			Expression(location, Type::Int(), Constant(10)));

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

	function.dump();

	Runner runner(&context, wrapFunctionWithModule(context, function));
	int result = 0;
	runner.run("main", result);
	EXPECT_EQ(result, 8);
}

TEST(LogicOps, orBooleansTrueOperands)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::lor>(
			location,
			Type::Bool(),
			Expression::trueExp(location),
			Expression::trueExp(location));

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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(LogicOps, orBooleansTrueAndFalseOperands)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::lor>(
			location,
			Type::Bool(),
			Expression::trueExp(location),
			Expression::falseExp(location));

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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(LogicOps, orBooleansFalseOperands)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::lor>(
			location,
			Type::Bool(),
			Expression::falseExp(location),
			Expression::falseExp(location));

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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(LogicOps, orIntegers)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::lor>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(9)),
			Expression(location, Type::Int(), Constant(10)));

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
	EXPECT_EQ(result, 11);
}
