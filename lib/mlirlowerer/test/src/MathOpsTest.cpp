#include <gtest/gtest.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/mlirlowerer/Runner.hpp>
#include <modelica/utils/SourceRange.hpp>

#include "TestUtils.hpp"

using namespace modelica;
using namespace std;

TEST(NegateOp, zeroInteger)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::negate,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 0));

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

TEST(NegateOp, positiveInteger)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::negate,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57));

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

TEST(NegateOp, negativeInteger)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::negate,
			Expression::constant(location, makeType<BuiltInType::Integer>(), -57));

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

TEST(NegateOp, zeroFloat)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::negate,
			Expression::constant(location, makeType<BuiltInType::Float>(), 0.0));

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

TEST(NegateOp, positiveFloat)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::negate,
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.0));

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

TEST(NegateOp, negativeFloat)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::negate,
			Expression::constant(location, makeType<BuiltInType::Float>(), -57.0));

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

TEST(AddOp, sameSignIntegers)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::add,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 3));

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

TEST(AddOp, differentSignIntegers)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::add,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2),
			Expression::constant(location, makeType<BuiltInType::Integer>(), -3));

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

TEST(AddOp, sameSignFloats)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::add,
			Expression::constant(location, makeType<BuiltInType::Float>(), 2.5),
			Expression::constant(location, makeType<BuiltInType::Float>(), 3.0));

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

TEST(AddOp, differentSignFloats)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::add,
			Expression::constant(location, makeType<BuiltInType::Float>(), 2.5),
			Expression::constant(location, makeType<BuiltInType::Float>(), -3.0));

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

TEST(AddOp, integerCastedToFloat)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::add,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2),
			Expression::constant(location, makeType<BuiltInType::Float>(), -3.5));

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

TEST(AddOp, multipleIntegers)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::add,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 3),
			Expression::constant(location, makeType<BuiltInType::Integer>(), -10),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 4));

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

TEST(AddOp, multipleFloats)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::add,
			Expression::constant(location, makeType<BuiltInType::Float>(), 2.3),
			Expression::constant(location, makeType<BuiltInType::Float>(), 3.1),
			Expression::constant(location, makeType<BuiltInType::Float>(), 4.9),
			Expression::constant(location, makeType<BuiltInType::Float>(), -2.4));

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

TEST(SubOp, sameSignIntegers)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::subtract,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 5),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 3));

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

TEST(SubOp, differentSignIntegers)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::subtract,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2),
			Expression::constant(location, makeType<BuiltInType::Integer>(), -3));

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

TEST(SubOp, sameSignFloats)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::subtract,
			Expression::constant(location, makeType<BuiltInType::Float>(), 2.7),
			Expression::constant(location, makeType<BuiltInType::Float>(), 3.4));

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

TEST(SubOp, differentSignFloats)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::subtract,
			Expression::constant(location, makeType<BuiltInType::Float>(), 2.3),
			Expression::constant(location, makeType<BuiltInType::Float>(), -3.4));

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

TEST(SubOp, integerCastedToFloat)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::subtract,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2),
			Expression::constant(location, makeType<BuiltInType::Float>(), -3.7));

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

TEST(SubOp, multipleIntegers)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::subtract,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 10),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 3),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 1));

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

TEST(SubOp, multipleFloats)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::subtract,
			Expression::constant(location, makeType<BuiltInType::Float>(), 10.7),
			Expression::constant(location, makeType<BuiltInType::Float>(), 3.2),
			Expression::constant(location, makeType<BuiltInType::Float>(), 2.4),
			Expression::constant(location, makeType<BuiltInType::Float>(), 1.5));

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

TEST(MulOp, sameSignIntegers)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::multiply,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 3));

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

TEST(MulOp, differentSignIntegers)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::multiply,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2),
			Expression::constant(location, makeType<BuiltInType::Integer>(), -3));

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

TEST(MulOp, sameSignFloats)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::multiply,
			Expression::constant(location, makeType<BuiltInType::Float>(), 2.3),
			Expression::constant(location, makeType<BuiltInType::Float>(), 3.7));

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

TEST(MulOp, differentSignFloats)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::multiply,
			Expression::constant(location, makeType<BuiltInType::Float>(), 2.3),
			Expression::constant(location, makeType<BuiltInType::Float>(), -3.7));

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

TEST(MulOp, integerCastedToFloat)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::multiply,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2),
			Expression::constant(location, makeType<BuiltInType::Float>(), -3.7));

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

TEST(MulOp, multipleIntegers)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::multiply,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 3),
			Expression::constant(location, makeType<BuiltInType::Integer>(), -10),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 4));

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

TEST(MulOp, multipleFloats)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::multiply,
			Expression::constant(location, makeType<BuiltInType::Float>(), 2.5F),
			Expression::constant(location, makeType<BuiltInType::Float>(), 3.7F),
			Expression::constant(location, makeType<BuiltInType::Float>(), 4.9F),
			Expression::constant(location, makeType<BuiltInType::Float>(), -2.0F));

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

TEST(DivOp, sameSignIntegers)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::divide,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 10),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 3));

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

TEST(DivOp, differentSignIntegers)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::divide,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 10),
			Expression::constant(location, makeType<BuiltInType::Integer>(), -3));

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

TEST(DivOp, sameSignFloats)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::divide,
			Expression::constant(location, makeType<BuiltInType::Float>(), 10.8),
			Expression::constant(location, makeType<BuiltInType::Float>(), 3.6));

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

TEST(DivOp, differentSignFloats)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::divide,
			Expression::constant(location, makeType<BuiltInType::Float>(), 10.8),
			Expression::constant(location, makeType<BuiltInType::Float>(), -3.6));

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

TEST(DivOp, integerCastedToFloat)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::divide,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 10),
			Expression::constant(location, makeType<BuiltInType::Float>(), -3.2));

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

TEST(DivOp, multipleIntegers)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::divide,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 120),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2),
			Expression::constant(location, makeType<BuiltInType::Integer>(), -3),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 4));

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

TEST(DivOp, multipleFloats)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::divide,
			Expression::constant(location, makeType<BuiltInType::Float>(), 120.4),
			Expression::constant(location, makeType<BuiltInType::Float>(), 3.2),
			Expression::constant(location, makeType<BuiltInType::Float>(), -8.6),
			Expression::constant(location, makeType<BuiltInType::Float>(), 2.5));

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

TEST(Pow, integerRaisedToInteger)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::powerOf,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 3));

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
	EXPECT_FLOAT_EQ(result, 8);
}

TEST(Pow, integerRaisedToFloat)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Integer>(),
			OperationKind::powerOf,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 4),
			Expression::constant(location, makeType<BuiltInType::Float>(), 0.5));

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
	EXPECT_FLOAT_EQ(result, 2);
}

TEST(Pow, floatRaisedToInteger)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::powerOf,
			Expression::constant(location, makeType<BuiltInType::Float>(), 2.5),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 2));

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
	EXPECT_FLOAT_EQ(result, 6.25);
}

TEST(Pow, floatRaisedToFloat)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Float>(),
			OperationKind::powerOf,
			Expression::constant(location, makeType<BuiltInType::Float>(), 1.5625),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 0.5));

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
	EXPECT_FLOAT_EQ(result, 1.25);
}
