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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::negate,
			Expression::constant(location, Type::Int(), 0));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::negate,
			Expression::constant(location, Type::Int(), 57));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::negate,
			Expression::constant(location, Type::Int(), -57));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::negate,
			Expression::constant(location, Type::Float(), 0.0));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::negate,
			Expression::constant(location, Type::Float(), 57.0));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::negate,
			Expression::constant(location, Type::Float(), -57.0));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::add,
			Expression::constant(location, Type::Int(), 2),
			Expression::constant(location, Type::Int(), 3));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::add,
			Expression::constant(location, Type::Int(), 2),
			Expression::constant(location, Type::Int(), -3));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::add,
			Expression::constant(location, Type::Float(), 2.5),
			Expression::constant(location, Type::Float(), 3.0));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::add,
			Expression::constant(location, Type::Float(), 2.5),
			Expression::constant(location, Type::Float(), -3.0));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::add,
			Expression::constant(location, Type::Int(), 2),
			Expression::constant(location, Type::Float(), -3.5));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::add,
			Expression::constant(location, Type::Int(), 2),
			Expression::constant(location, Type::Int(), 3),
			Expression::constant(location, Type::Int(), -10),
			Expression::constant(location, Type::Int(), 4));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::add,
			Expression::constant(location, Type::Float(), 2.3),
			Expression::constant(location, Type::Float(), 3.1),
			Expression::constant(location, Type::Float(), 4.9),
			Expression::constant(location, Type::Float(), -2.4));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::subtract,
			Expression::constant(location, Type::Int(), 5),
			Expression::constant(location, Type::Int(), 3));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::subtract,
			Expression::constant(location, Type::Int(), 2),
			Expression::constant(location, Type::Int(), -3));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::subtract,
			Expression::constant(location, Type::Float(), 2.7),
			Expression::constant(location, Type::Float(), 3.4));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::subtract,
			Expression::constant(location, Type::Float(), 2.3),
			Expression::constant(location, Type::Float(), -3.4));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::subtract,
			Expression::constant(location, Type::Int(), 2),
			Expression::constant(location, Type::Float(), -3.7));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::subtract,
			Expression::constant(location, Type::Int(), 10),
			Expression::constant(location, Type::Int(), 3),
			Expression::constant(location, Type::Int(), 2),
			Expression::constant(location, Type::Int(), 1));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::subtract,
			Expression::constant(location, Type::Float(), 10.7),
			Expression::constant(location, Type::Float(), 3.2),
			Expression::constant(location, Type::Float(), 2.4),
			Expression::constant(location, Type::Float(), 1.5));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::multiply,
			Expression::constant(location, Type::Int(), 2),
			Expression::constant(location, Type::Int(), 3));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::multiply,
			Expression::constant(location, Type::Int(), 2),
			Expression::constant(location, Type::Int(), -3));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::multiply,
			Expression::constant(location, Type::Float(), 2.3),
			Expression::constant(location, Type::Float(), 3.7));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::multiply,
			Expression::constant(location, Type::Float(), 2.3),
			Expression::constant(location, Type::Float(), -3.7));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::multiply,
			Expression::constant(location, Type::Int(), 2),
			Expression::constant(location, Type::Float(), -3.7));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::multiply,
			Expression::constant(location, Type::Int(), 2),
			Expression::constant(location, Type::Int(), 3),
			Expression::constant(location, Type::Int(), -10),
			Expression::constant(location, Type::Int(), 4));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::multiply,
			Expression::constant(location, Type::Float(), 2.5F),
			Expression::constant(location, Type::Float(), 3.7F),
			Expression::constant(location, Type::Float(), 4.9F),
			Expression::constant(location, Type::Float(), -2.0F));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::divide,
			Expression::constant(location, Type::Int(), 10),
			Expression::constant(location, Type::Int(), 3));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::divide,
			Expression::constant(location, Type::Int(), 10),
			Expression::constant(location, Type::Int(), -3));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::divide,
			Expression::constant(location, Type::Float(), 10.8),
			Expression::constant(location, Type::Float(), 3.6));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::divide,
			Expression::constant(location, Type::Float(), 10.8),
			Expression::constant(location, Type::Float(), -3.6));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::divide,
			Expression::constant(location, Type::Int(), 10),
			Expression::constant(location, Type::Float(), -3.2));

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

	Expression expression = Expression::operation(
			location,
			Type::Int(),
			OperationKind::divide,
			Expression::constant(location, Type::Int(), 120),
			Expression::constant(location, Type::Int(), 2),
			Expression::constant(location, Type::Int(), -3),
			Expression::constant(location, Type::Int(), 4));

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

	Expression expression = Expression::operation(
			location,
			Type::Float(),
			OperationKind::divide,
			Expression::constant(location, Type::Float(), 120.4),
			Expression::constant(location, Type::Float(), 3.2),
			Expression::constant(location, Type::Float(), -8.6),
			Expression::constant(location, Type::Float(), 2.5));

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
