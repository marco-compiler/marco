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

TEST(MathOps, negateZeroInteger)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::negate>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(0)));

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
	EXPECT_EQ(result, 0);
}

TEST(MathOps, negatePositiveInteger)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::negate>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(57)));

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
	EXPECT_EQ(result, -57);
}

TEST(MathOps, negateNegativeInteger)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::negate>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(-57)));

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
	EXPECT_EQ(result, 57);
}

TEST(MathOps, negateZeroFloat)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::negate>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(0.0)));

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
	EXPECT_FLOAT_EQ(result, 0);
}

TEST(MathOps, negatePositiveFloat)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::negate>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(57.0)));

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
	EXPECT_FLOAT_EQ(result, -57.0);
}

TEST(MathOps, negateNegativeFloat)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::negate>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(-57.0)));

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
	EXPECT_FLOAT_EQ(result, 57.0);
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
			Expression(location, Type::Float(), Constant(2.5)),
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
	EXPECT_FLOAT_EQ(result, 5.5);
}

TEST(MathOps, addDifferentSignFloats)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::add>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(2.5)),
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
	EXPECT_FLOAT_EQ(result, -0.5);
}

TEST(MathOps, addIntegerCastedToFloat)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::add>(
			location,
			Type::Float(),
			Expression(location, Type::Int(), Constant(2)),
			Expression(location, Type::Float(), Constant(-3.5)));

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
	EXPECT_FLOAT_EQ(result, -1.5);
}

TEST(MathOps, addMultipleIntegers)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::add>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(2)),
			Expression(location, Type::Int(), Constant(3)),
			Expression(location, Type::Int(), Constant(-10)),
			Expression(location, Type::Int(), Constant(4)));

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

TEST(MathOps, addMultipleFloats)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::add>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(2.3)),
			Expression(location, Type::Float(), Constant(3.1)),
			Expression(location, Type::Float(), Constant(4.9)),
			Expression(location, Type::Float(), Constant(-2.4)));

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
	EXPECT_FLOAT_EQ(result, 7.9);
}

TEST(MathOps, subSameSignIntegers)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::subtract>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(5)),
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
	EXPECT_EQ(result, 2);
}

TEST(MathOps, subDifferentSignIntegers)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::subtract>(
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
	EXPECT_EQ(result, 5);
}

TEST(MathOps, subSameSignFloats)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::subtract>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(2.7)),
			Expression(location, Type::Float(), Constant(3.4)));

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
	EXPECT_FLOAT_EQ(result, -0.7);
}

TEST(MathOps, subDifferentSignFloats)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::subtract>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(2.3)),
			Expression(location, Type::Float(), Constant(-3.4)));

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
	EXPECT_FLOAT_EQ(result, 5.7);
}

TEST(MathOps, subIntegerCastedToFloat)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::subtract>(
			location,
			Type::Float(),
			Expression(location, Type::Int(), Constant(2)),
			Expression(location, Type::Float(), Constant(-3.7)));

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
	EXPECT_FLOAT_EQ(result, 5.7);
}

TEST(MathOps, subMultipleIntegers)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::subtract>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(10)),
			Expression(location, Type::Int(), Constant(3)),
			Expression(location, Type::Int(), Constant(2)),
			Expression(location, Type::Int(), Constant(1)));

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
	EXPECT_EQ(result, 4);
}

TEST(MathOps, subMultipleFloats)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::subtract>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(10.7)),
			Expression(location, Type::Float(), Constant(3.2)),
			Expression(location, Type::Float(), Constant(2.4)),
			Expression(location, Type::Float(), Constant(1.5)));

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
	EXPECT_FLOAT_EQ(result, 3.6);
}

TEST(MathOps, mulSameSignIntegers)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::multiply>(
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
	EXPECT_EQ(result, 6);
}

TEST(MathOps, mulDifferentSignIntegers)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::multiply>(
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
	EXPECT_EQ(result, -6);
}

TEST(MathOps, mulSameSignFloats)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::multiply>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(2.3)),
			Expression(location, Type::Float(), Constant(3.7)));

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
	EXPECT_FLOAT_EQ(result, 8.51);
}

TEST(MathOps, mulDifferentSignFloats)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::multiply>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(2.3)),
			Expression(location, Type::Float(), Constant(-3.7)));

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
	EXPECT_FLOAT_EQ(result, -8.51);
}

TEST(MathOps, mulIntegerCastedToFloat)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::multiply>(
			location,
			Type::Float(),
			Expression(location, Type::Int(), Constant(2)),
			Expression(location, Type::Float(), Constant(-3.7)));

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
	EXPECT_FLOAT_EQ(result, -7.4);
}

TEST(MathOps, mulMultipleIntegers)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::multiply>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(2)),
			Expression(location, Type::Int(), Constant(3)),
			Expression(location, Type::Int(), Constant(-10)),
			Expression(location, Type::Int(), Constant(4)));

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
	EXPECT_EQ(result, -240);
}

TEST(MathOps, mulMultipleFloats)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::multiply>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(2.5F)),
			Expression(location, Type::Float(), Constant(3.7F)),
			Expression(location, Type::Float(), Constant(4.9F)),
			Expression(location, Type::Float(), Constant(-2.0F)));

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
	EXPECT_FLOAT_EQ(result, -90.65);
}

TEST(MathOps, divSameSignIntegers)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::divide>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(10)),
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
	EXPECT_EQ(result, 3);
}

TEST(MathOps, divDifferentSignIntegers)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::divide>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(10)),
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
	EXPECT_EQ(result, -3);
}

TEST(MathOps, divSameSignFloats)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::divide>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(10.8)),
			Expression(location, Type::Float(), Constant(3.6)));

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
	EXPECT_FLOAT_EQ(result, 3.0);
}

TEST(MathOps, divDifferentSignFloats)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::divide>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(10.8)),
			Expression(location, Type::Float(), Constant(-3.6)));

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
	EXPECT_FLOAT_EQ(result, -3.0);
}

TEST(MathOps, divIntegerCastedToFloat)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::divide>(
			location,
			Type::Float(),
			Expression(location, Type::Int(), Constant(10)),
			Expression(location, Type::Float(), Constant(-3.2)));

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
	EXPECT_FLOAT_EQ(result, -3.125);
}

TEST(MathOps, divMultipleIntegers)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::divide>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(120)),
			Expression(location, Type::Int(), Constant(2)),
			Expression(location, Type::Int(), Constant(-3)),
			Expression(location, Type::Int(), Constant(4)));

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
	EXPECT_EQ(result, -5);
}

TEST(MathOps, divMultipleFloats)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::divide>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(120.4)),
			Expression(location, Type::Float(), Constant(3.2)),
			Expression(location, Type::Float(), Constant(-8.6)),
			Expression(location, Type::Float(), Constant(2.5)));

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
	EXPECT_FLOAT_EQ(result, -1.75);
}
