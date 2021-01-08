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

TEST(AndOp, booleansTrueOperands)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::land,
			Expression::constant(location, makeType<BuiltInType::Boolean>(), true),
			Expression::constant(location, makeType<BuiltInType::Boolean>(), true));

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

TEST(AndOp, booleansTrueAndFalseOperands)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::land,
			Expression::constant(location, makeType<BuiltInType::Boolean>(), true),
			Expression::constant(location, makeType<BuiltInType::Boolean>(), false));

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

TEST(AndOp, booleansFalseOperands)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::land,
			Expression::constant(location, makeType<BuiltInType::Boolean>(), false),
			Expression::constant(location, makeType<BuiltInType::Boolean>(), false));

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

TEST(AndOp, integers)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::land,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 13),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 10));

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

TEST(OrOp, booleansTrueOperands)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::lor,
			Expression::constant(location, makeType<BuiltInType::Boolean>(), true),
			Expression::constant(location, makeType<BuiltInType::Boolean>(), true));

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

TEST(OrOp, booleansTrueAndFalseOperands)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::lor,
			Expression::constant(location, makeType<BuiltInType::Boolean>(), true),
			Expression::constant(location, makeType<BuiltInType::Boolean>(), false));

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

TEST(OrOp, booleansFalseOperands)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::lor,
			Expression::constant(location, makeType<BuiltInType::Boolean>(), false),
			Expression::constant(location, makeType<BuiltInType::Boolean>(), false));

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

TEST(OrOp, integers)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::lor,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 9),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 10));

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
