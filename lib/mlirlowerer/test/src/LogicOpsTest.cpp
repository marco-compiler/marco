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

	Expression expression = Expression::operation(
			location,
			Type::Bool(),
			OperationKind::land,
			Expression::constant(location, Type::Bool(), true),
			Expression::constant(location, Type::Bool(), true));

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

	Expression expression = Expression::operation(
			location,
			Type::Bool(),
			OperationKind::land,
			Expression::constant(location, Type::Bool(), true),
			Expression::constant(location, Type::Bool(), false));

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

	Expression expression = Expression::operation(
			location,
			Type::Bool(),
			OperationKind::land,
			Expression::constant(location, Type::Bool(), false),
			Expression::constant(location, Type::Bool(), false));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::land,
			Expression::constant(location, Type::Int(), 13),
			Expression::constant(location, Type::Int(), 10));

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
	EXPECT_EQ(result, 8);
}

TEST(LogicOps, orBooleansTrueOperands)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::operation(
			location,
			Type::Bool(),
			OperationKind::lor,
			Expression::constant(location, Type::Bool(), true),
			Expression::constant(location, Type::Bool(), true));

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

	Expression expression = Expression::operation(
			location,
			Type::Bool(),
			OperationKind::lor,
			Expression::constant(location, Type::Bool(), true),
			Expression::constant(location, Type::Bool(), false));

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

	Expression expression = Expression::operation(
			location,
			Type::Bool(),
			OperationKind::lor,
			Expression::constant(location, Type::Bool(), false),
			Expression::constant(location, Type::Bool(), false));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::lor,
			Expression::constant(location, Type::Int(), 9),
			Expression::constant(location, Type::Int(), 10));

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
