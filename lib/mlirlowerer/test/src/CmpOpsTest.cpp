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

TEST(CmpOps, eqIntTrue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::equal>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(57)),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(CmpOps, eqIntFalse)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::equal>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(23)),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(CmpOps, eqFloatsTrue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::equal>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(57.23)),
			Expression(location, Type::Float(), Constant(57.23)));

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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(CmpOps, eqFloatsFalse)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::equal>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(23.57)),
			Expression(location, Type::Float(), Constant(57.23)));

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

TEST(CmpOps, eqIntCastedToFloatTrue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::equal>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(57)),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(CmpOps, eqIntCastedToFloatFalse)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::equal>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(23)),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(CmpOps, notEqIntTrue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::different>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(23)),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(CmpOps, notEqIntFalse)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::different>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(57)),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(CmpOps, notEqFloatsTrue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::different>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(23.57)),
			Expression(location, Type::Float(), Constant(57.23)));

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

TEST(CmpOps, notEqFloatsFalse)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::different>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(57.23)),
			Expression(location, Type::Float(), Constant(57.23)));

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

TEST(CmpOps, notEqIntCastedToFloatTrue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::different>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(23)),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(CmpOps, notEqIntCastedToFloatFalse)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::different>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(57)),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(CmpOps, gtIntTrue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greater>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(57)),
			Expression(location, Type::Int(), Constant(23)));

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

TEST(CmpOps, gtIntFalseSameValue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greater>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(23)),
			Expression(location, Type::Int(), Constant(23)));

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

TEST(CmpOps, gtIntFalseDifferentValue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greater>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(23)),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(CmpOps, gtFloatsTrue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greater>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(57.23)),
			Expression(location, Type::Float(), Constant(23.57)));

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

TEST(CmpOps, gtFloatsFalseSameValues)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greater>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(57.23)),
			Expression(location, Type::Float(), Constant(57.23)));

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

TEST(CmpOps, gtFloatsFalseDifferentValues)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greater>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(23.57)),
			Expression(location, Type::Float(), Constant(57.23)));

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

TEST(CmpOps, gtIntCastedToFloatTrue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greater>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(57)),
			Expression(location, Type::Float(), Constant(23.0)));

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

TEST(CmpOps, gtIntCastedToFloatFalse)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greater>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(23)),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(CmpOps, gteIntTrueSameValues)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greaterEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(57)),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(CmpOps, gteIntTrueDifferentValues)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greaterEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(57)),
			Expression(location, Type::Int(), Constant(23)));

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

TEST(CmpOps, gteIntFalse)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greaterEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(23)),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(CmpOps, gteFloatsTrueSameValues)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greaterEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(57.23)),
			Expression(location, Type::Float(), Constant(57.23)));

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

TEST(CmpOps, gteFloatsTrueDifferentValues)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greaterEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(57.23)),
			Expression(location, Type::Float(), Constant(23.57)));

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

TEST(CmpOps, gteFloatsFalse)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greaterEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(23.57)),
			Expression(location, Type::Float(), Constant(57.23)));

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

TEST(CmpOps, gteIntCastedToFloatTrue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greaterEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(57)),
			Expression(location, Type::Float(), Constant(23.0)));

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

TEST(CmpOps, gteIntCastedToFloatFalse)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::greaterEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(23)),
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
	bool result = true;
	runner.run("main", result);
	EXPECT_FALSE(result);
}

TEST(CmpOps, ltIntTrue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::less>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(23)),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(CmpOps, ltIntFalseSameValue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::less>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(23)),
			Expression(location, Type::Int(), Constant(23)));

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

TEST(CmpOps, ltIntFalseDifferentValue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::less>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(57)),
			Expression(location, Type::Int(), Constant(23)));

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

TEST(CmpOps, ltFloatsTrue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::less>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(23.57)),
			Expression(location, Type::Float(), Constant(57.23)));

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

TEST(CmpOps, ltFloatsFalseSameValues)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::less>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(57.23)),
			Expression(location, Type::Float(), Constant(57.23)));

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

TEST(CmpOps, ltFloatsFalseDifferentValues)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::less>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(57.23)),
			Expression(location, Type::Float(), Constant(23.57)));

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

TEST(CmpOps, ltIntCastedToFloatTrue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::less>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(23)),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(CmpOps, ltIntCastedToFloatFalse)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::less>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(57)),
			Expression(location, Type::Float(), Constant(23.0)));

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

TEST(CmpOps, lteIntTrueSameValues)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::lessEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(57)),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(CmpOps, lteIntTrueDifferentValues)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::lessEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(23)),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(CmpOps, lteIntFalse)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::lessEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(57)),
			Expression(location, Type::Int(), Constant(23)));

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

TEST(CmpOps, lteFloatsTrueSameValues)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::lessEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(57.23)),
			Expression(location, Type::Float(), Constant(57.23)));

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

TEST(CmpOps, lteFloatsTrueDifferentValues)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::lessEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(23.57)),
			Expression(location, Type::Float(), Constant(57.23)));

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

TEST(CmpOps, lteFloatsFalse)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::lessEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Float(), Constant(57.23)),
			Expression(location, Type::Float(), Constant(23.57)));

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

TEST(CmpOps, lteIntCastedToFloatTrue)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::lessEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(23)),
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
	bool result = false;
	runner.run("main", result);
	EXPECT_TRUE(result);
}

TEST(CmpOps, lteIntCastedToFloatFalse)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::lessEqual>(
			location,
			Type::Bool(),
			Expression(location, Type::Int(), Constant(57)),
			Expression(location, Type::Float(), Constant(23.0)));

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
