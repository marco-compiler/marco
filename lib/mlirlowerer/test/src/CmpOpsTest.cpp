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

TEST(EqOp, intTrue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::equal,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(EqOp, intFalse)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::equal,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(EqOp, floatsTrue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::equal,
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23),
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23));

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

TEST(EqOp, floatsFalse)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::equal,
			Expression::constant(location, makeType<BuiltInType::Float>(), 23.57),
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23));

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

TEST(EqOp, intCastedToFloatTrue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::equal,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(EqOp, intCastedToFloatFalse)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::equal,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(NotEqOp, intTrue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::different,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(NotEqOp, intFalse)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::different,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(NotEqOp, floatsTrue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::different,
			Expression::constant(location, makeType<BuiltInType::Float>(), 23.57),
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23));

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

TEST(NotEqOp, floatsFalse)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::different,
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23),
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23));

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

TEST(NotEqOp, intCastedToFloatTrue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::different,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(NotEqOp, intCastedToFloatFalse)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::different,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(GtOp, intTrue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greater,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23));

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

TEST(GtOp, intFalseSameValue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greater,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23));

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

TEST(GtOp, intFalseDifferentValue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greater,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(GtOp, floatsTrue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greater,
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23),
			Expression::constant(location, makeType<BuiltInType::Float>(), 23.57));

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

TEST(GtOp, floatsFalseSameValues)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greater,
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23),
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23));

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

TEST(GtOp, floatsFalseDifferentValues)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greater,
			Expression::constant(location, makeType<BuiltInType::Float>(), 23.57),
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23));

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

TEST(GtOp, intCastedToFloatTrue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greater,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57),
			Expression::constant(location, makeType<BuiltInType::Float>(), 23.0));

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

TEST(GtOp, intCastedToFloatFalse)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greater,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(GteOp, intTrueSameValues)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greaterEqual,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(GteOp, intTrueDifferentValues)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greaterEqual,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23));

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

TEST(GteOp, intFalse)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greaterEqual,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(GteOp, floatsTrueSameValues)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greaterEqual,
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23),
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23));

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

TEST(GteOp, floatsTrueDifferentValues)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greaterEqual,
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23),
			Expression::constant(location, makeType<BuiltInType::Float>(), 23.57));

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

TEST(GteOp, floatsFalse)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greaterEqual,
			Expression::constant(location, makeType<BuiltInType::Float>(), 23.57),
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23));

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

TEST(GteOp, intCastedToFloatTrue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greaterEqual,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57),
			Expression::constant(location, makeType<BuiltInType::Float>(), 23.0));

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

TEST(GteOp, intCastedToFloatFalse)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::greaterEqual,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(LtOp, intTrue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::less,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(LtOp, intFalseSameValue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::less,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23));

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

TEST(LtOp, intFalseDifferentValue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::less,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23));

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

TEST(LtOp, floatsTrue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::less,
			Expression::constant(location, makeType<BuiltInType::Float>(), 23.57),
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23));

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

TEST(LtOp, floatsFalseSameValues)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::less,
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23),
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23));

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

TEST(LtOp, floatsFalseDifferentValues)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::less,
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23),
			Expression::constant(location, makeType<BuiltInType::Float>(), 23.57));

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

TEST(LtOp, intCastedToFloatTrue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::less,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(LtOp, intCastedToFloatFalse)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::less,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57),
			Expression::constant(location, makeType<BuiltInType::Float>(), 23.0));

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

TEST(LteOp, intTrueSameValues)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::lessEqual,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(LteOp, intTrueDifferentValues)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::lessEqual,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(LteOp, intFalse)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::lessEqual,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23));

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

TEST(LteOp, floatsTrueSameValues)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::lessEqual,
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23),
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23));

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

TEST(LteOp, floatsTrueDifferentValues)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::lessEqual,
			Expression::constant(location, makeType<BuiltInType::Float>(), 23.57),
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23));

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

TEST(LteOp, floatsFalse)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::lessEqual,
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.23),
			Expression::constant(location, makeType<BuiltInType::Float>(), 23.57));

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

TEST(LteOp, intCastedToFloatTrue)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::lessEqual,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(LteOp, intCastedToFloatFalse)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<BuiltInType::Boolean>(),
			OperationKind::lessEqual,
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57),
			Expression::constant(location, makeType<BuiltInType::Float>(), 23.0));

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
