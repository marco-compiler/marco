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
	EXPECT_EQ(result, 5.5);
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
	EXPECT_EQ(result, -0.5);
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
	EXPECT_EQ(result, -1.5);
}

TEST(MathOps, addFloatCastedToInt)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::add>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(2)),
			Expression(location, Type::Float(), Constant(-1.5)));

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
 // TODO: add more values
	Expression expression = Expression::op<OperationKind::add>(
			location,
			Type::Float(),
			Expression(location, Type::Float(), Constant(2.3F)),
			Expression(location, Type::Float(), Constant(3.1F)));

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
	float difference = result - 5.4F;
	llvm::errs() << "difference: " << difference;
	EXPECT_EQ(result, 5.4F);
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
	EXPECT_EQ(result, -0.7);
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
	EXPECT_EQ(result, 5.7);
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
	EXPECT_EQ(result, 5.7);
}

TEST(MathOps, subFloatCastedToInt)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::subtract>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(2)),
			Expression(location, Type::Float(), Constant(-1.5)));

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
	EXPECT_EQ(result, 3.0);
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
	EXPECT_EQ(result, 3.6);
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
	EXPECT_EQ(result, 8.51);
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
	EXPECT_EQ(result, -8.51);
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
	EXPECT_EQ(result, -7.4);
}

TEST(MathOps, mulFloatCastedToInt)	 // NOLINT
{
	SourcePosition location("-", 0, 0);

	Expression expression = Expression::op<OperationKind::multiply>(
			location,
			Type::Int(),
			Expression(location, Type::Int(), Constant(2)),
			Expression(location, Type::Float(), Constant(-2.5)));

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
	EXPECT_EQ(result, -5.0);
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
			Expression(location, Type::Float(), Constant(2.5)),
			Expression(location, Type::Float(), Constant(3.7)),
			Expression(location, Type::Float(), Constant(-10.2)),
			Expression(location, Type::Float(), Constant(4.9)));

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
	EXPECT_EQ(result, -144.33);
}
