#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/mlirlowerer/Runner.hpp>
#include <modelica/utils/SourceRange.hpp>
#include "TestUtils.hpp"

using namespace modelica;
using namespace std;

TEST(AddOp, scalarIntegers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Integer z;
	 *
	 *   algorithm
	 *     z := x + y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::add,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
														Expression::reference(location, makeType<BuiltInType::Integer>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<int, 2> xData = { 23, 57 };
	array<int, 2> yData = { 57, -23 };
	array<int, 2> zData = { 0, 0 };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		int y = get<1>(tuple);
		int z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x + y);
	}
}

TEST(AddOp, vectorIntegers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[3] x;
	 *   input Integer[3] y;
	 *   output Integer[3] z;
	 *
	 *   algorithm
	 *     z := x + y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Integer>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(3), "z"),
			Expression::operation(location, makeType<BuiltInType::Integer>(3), OperationKind::add,
														Expression::reference(location, makeType<BuiltInType::Integer>(3), "x"),
														Expression::reference(location, makeType<BuiltInType::Integer>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	module.dump();

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 0, 0, 0 };

	int* xPtr = x.data();
	int* yPtr = y.data();
	int* zPtr = z.data();

	runner.run("main", xPtr, yPtr, zPtr);

	for (const auto& tuple : llvm::zip(x, y, z))
		EXPECT_EQ(get<2>(tuple), get<0>(tuple) + get<1>(tuple));
}

TEST(AddOp, scalarFloats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Real z;
	 *
	 *   algorithm
	 *     z := x + y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Float>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Float>(), OperationKind::add,
														Expression::reference(location, makeType<BuiltInType::Float>(), "x"),
														Expression::reference(location, makeType<BuiltInType::Float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	module.dump();

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<float, 2> xData = { 23.2f, 57.5f };
	array<float, 2> yData = { 57.3f, -23.7f };
	array<float, 2> zData = { 0.0f, 0.0f };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		float x = get<0>(tuple);
		float y = get<1>(tuple);
		float z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_FLOAT_EQ(z, x + y);
	}
}

TEST(AddOp, vectorFloats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real[3] x;
	 *   input Real[3] y;
	 *   output Real[3] z;
	 *
	 *   algorithm
	 *     z := x + y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Float>(3), "z"),
			Expression::operation(location, makeType<BuiltInType::Float>(3), OperationKind::add,
														Expression::reference(location, makeType<BuiltInType::Float>(3), "x"),
														Expression::reference(location, makeType<BuiltInType::Float>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<float, 3> x = { 10.1f, 23.3f, 57.8f };
	array<float, 3> y = { 10.2f, 57.3f, -23.5f };
	array<float, 3> z = { 0.0f, 0.0f, 0.0f };

	float* xPtr = x.data();
	float* yPtr = y.data();
	float* zPtr = z.data();

	runner.run("main", xPtr, yPtr, zPtr);

	for (const auto& tuple : llvm::zip(x, y, z))
		EXPECT_FLOAT_EQ(get<2>(tuple), get<0>(tuple) + get<1>(tuple));
}

TEST(AddOp, integerCastedToFloatScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Real z;
	 *
	 *   algorithm
	 *     z := x + y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Float>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Float>(), OperationKind::add,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
														Expression::reference(location, makeType<BuiltInType::Float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	module.dump();

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<int, 3> xData = { 2, -3, -3 };
	array<float, 3> yData = { -3.5f, 5.2f, -2.0f };
	array<float, 3> zData = { 0.0f, 0.0f, 0.0f };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		float y = get<1>(tuple);
		float z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_FLOAT_EQ(z, x + y);
	}
}

TEST(AddOp, integerCastedToFloatVector)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[3] x;
	 *   input Real[3] y;
	 *   output Real[3] z;
	 *
	 *   algorithm
	 *     z := x + y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Float>(3), "z"),
			Expression::operation(location, makeType<BuiltInType::Float>(3), OperationKind::add,
														Expression::reference(location, makeType<BuiltInType::Integer>(3), "x"),
														Expression::reference(location, makeType<BuiltInType::Float>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	module.dump();

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<int, 3> x = { 2, -3, -3 };
	array<float, 3> y = { -3.5f, 5.2f, -2.0f };
	array<float, 3> z = { 0.0f, 0.0f, 0.0f };

	int* xPtr = x.data();
	float* yPtr = y.data();
	float* zPtr = z.data();

	runner.run("main", xPtr, yPtr, zPtr);

	for (const auto& tuple : llvm::zip(x, y, z))
		EXPECT_FLOAT_EQ(get<2>(tuple), get<0>(tuple) + get<1>(tuple));
}

TEST(AddOp, multipleValues)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   input Integer z;
	 *   output Integer t;
	 *
	 *   algorithm
	 *     t := x + y + z;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member tMember(location, "t", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "t"),
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::add,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
														Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
														Expression::reference(location, makeType<BuiltInType::Integer>(), "z")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember, tMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<int, 3> xData = { 10, 23, 57 };
	array<int, 3> yData = { 10, 57, -23 };
	array<int, 3> zData = { 4, -7, -15 };
	array<int, 3> tData = { 0, 0, 0 };

	for (const auto& tuple : llvm::zip(xData, yData, zData, tData))
	{
		int x = get<0>(tuple);
		int y = get<1>(tuple);
		int z = get<2>(tuple);
		int t = get<3>(tuple);

		runner.run("main", x, y, z, t);

		EXPECT_EQ(t, x + y + z);
	}
}

TEST(SubOp, scalarIntegers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Integer z;
	 *
	 *   algorithm
	 *     z := x - y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subtract,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
														Expression::reference(location, makeType<BuiltInType::Integer>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<int, 2> xData = { 23, 57 };
	array<int, 2> yData = { 57, -23 };
	array<int, 2> zData = { 0, 0 };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		int y = get<1>(tuple);
		int z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x - y);
	}
}

TEST(SubOp, vectorIntegers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[3] x;
	 *   input Integer[3] y;
	 *   output Integer[3] z;
	 *
	 *   algorithm
	 *     z := x - y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Integer>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(3), "z"),
			Expression::operation(location, makeType<BuiltInType::Integer>(3), OperationKind::subtract,
														Expression::reference(location, makeType<BuiltInType::Integer>(3), "x"),
														Expression::reference(location, makeType<BuiltInType::Integer>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	module.dump();

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 0, 0, 0 };

	int* xPtr = x.data();
	int* yPtr = y.data();
	int* zPtr = z.data();

	runner.run("main", xPtr, yPtr, zPtr);

	for (const auto& tuple : llvm::zip(x, y, z))
		EXPECT_EQ(get<2>(tuple), get<0>(tuple) - get<1>(tuple));
}

TEST(SubOp, scalarFloats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Real z;
	 *
	 *   algorithm
	 *     z := x - y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Float>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Float>(), OperationKind::subtract,
														Expression::reference(location, makeType<BuiltInType::Float>(), "x"),
														Expression::reference(location, makeType<BuiltInType::Float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<float, 2> xData = { 23.2f, 57.5f };
	array<float, 2> yData = { 57.3f, -23.7f };
	array<float, 2> zData = { 0.0f, 0.0f };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		float x = get<0>(tuple);
		float y = get<1>(tuple);
		float z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_FLOAT_EQ(z, x - y);
	}
}

TEST(SubOp, vectorFloats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real[3] x;
	 *   input Real[3] y;
	 *   output Real[3] z;
	 *
	 *   algorithm
	 *     z := x - y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Float>(3), "z"),
			Expression::operation(location, makeType<BuiltInType::Float>(3), OperationKind::subtract,
														Expression::reference(location, makeType<BuiltInType::Float>(3), "x"),
														Expression::reference(location, makeType<BuiltInType::Float>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<float, 3> x = { 10.1f, 23.3f, 57.8f };
	array<float, 3> y = { 10.2f, 57.3f, -23.5f };
	array<float, 3> z = { 0.0f, 0.0f, 0.0f };

	float* xPtr = x.data();
	float* yPtr = y.data();
	float* zPtr = z.data();

	runner.run("main", xPtr, yPtr, zPtr);

	for (const auto& tuple : llvm::zip(x, y, z))
		EXPECT_FLOAT_EQ(get<2>(tuple), get<0>(tuple) - get<1>(tuple));
}

TEST(SubOp, integerCastedToFloatScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Real z;
	 *
	 *   algorithm
	 *     z := x - y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Float>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Float>(), OperationKind::subtract,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
														Expression::reference(location, makeType<BuiltInType::Float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<int, 3> xData = { 2, -3, -3 };
	array<float, 3> yData = { -3.5f, 5.2f, -2.0f };
	array<float, 3> zData = { 0.0f, 0.0f, 0.0f };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		float y = get<1>(tuple);
		float z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_FLOAT_EQ(z, x - y);
	}
}

TEST(SubOp, integerCastedToFloatVector)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[3] x;
	 *   input Real[3] y;
	 *   output Real[3] z;
	 *
	 *   algorithm
	 *     z := x - y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Float>(3), "z"),
			Expression::operation(location, makeType<BuiltInType::Float>(3), OperationKind::subtract,
														Expression::reference(location, makeType<BuiltInType::Integer>(3), "x"),
														Expression::reference(location, makeType<BuiltInType::Float>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<int, 3> x = { 2, -3, -3 };
	array<float, 3> y = { -3.5f, 5.2f, -2.0f };
	array<float, 3> z = { 0.0f, 0.0f, 0.0f };

	int* xPtr = x.data();
	float* yPtr = y.data();
	float* zPtr = z.data();

	runner.run("main", xPtr, yPtr, zPtr);

	for (const auto& tuple : llvm::zip(x, y, z))
		EXPECT_FLOAT_EQ(get<2>(tuple), get<0>(tuple) - get<1>(tuple));
}

TEST(SubOp, multipleValues)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   input Integer z;
	 *   output Integer t;
	 *
	 *   algorithm
	 *     t := x - y - z;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member tMember(location, "t", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "t"),
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subtract,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
														Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
														Expression::reference(location, makeType<BuiltInType::Integer>(), "z")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember, tMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<int, 3> xData = { 10, 23, 57 };
	array<int, 3> yData = { 10, 57, -23 };
	array<int, 3> zData = { 4, -7, -15 };
	array<int, 3> tData = { 0, 0, 0 };

	for (const auto& tuple : llvm::zip(xData, yData, zData, tData))
	{
		int x = get<0>(tuple);
		int y = get<1>(tuple);
		int z = get<2>(tuple);
		int t = get<3>(tuple);

		runner.run("main", x, y, z, t);

		EXPECT_EQ(t, x - y - z);
	}
}

TEST(MulOp, scalarIntegers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Integer z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::multiply,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
														Expression::reference(location, makeType<BuiltInType::Integer>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<int, 2> xData = { 2, 5 };
	array<int, 2> yData = { 3, -3 };
	array<int, 2> zData = { 0, 0 };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		int y = get<1>(tuple);
		int z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x * y);
	}
}

TEST(MulOp, scalarFloats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Real z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Float>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Float>(), OperationKind::multiply,
														Expression::reference(location, makeType<BuiltInType::Float>(), "x"),
														Expression::reference(location, makeType<BuiltInType::Float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<float, 2> xData = { 2.3f, 5.7f };
	array<float, 2> yData = { 23.57f, -23.57f };
	array<float, 2> zData = { 0.0f, 0.0f };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		float x = get<0>(tuple);
		float y = get<1>(tuple);
		float z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_FLOAT_EQ(z, x * y);
	}
}

TEST(MulOp, integerCastedToFloatScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Real z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Float>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Float>(), OperationKind::multiply,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
														Expression::reference(location, makeType<BuiltInType::Float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<int, 3> xData = { 2, -3, -3 };
	array<float, 3> yData = { -3.5f, 5.2f, -2.0f };
	array<float, 3> zData = { 0.0f, 0.0f, 0.0f };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		float y = get<1>(tuple);
		float z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_FLOAT_EQ(z, x * y);
	}
}

TEST(MulOp, multipleValues)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   input Integer z;
	 *   output Integer t;
	 *
	 *   algorithm
	 *     t := x * y * z;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member tMember(location, "t", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "t"),
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::multiply,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
														Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
														Expression::reference(location, makeType<BuiltInType::Integer>(), "z")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember, tMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<int, 3> xData = { 10, 23, 57 };
	array<int, 3> yData = { 10, 57, -23 };
	array<int, 3> zData = { 4, -7, -15 };
	array<int, 3> tData = { 0, 0, 0 };

	for (const auto& tuple : llvm::zip(xData, yData, zData, tData))
	{
		int x = get<0>(tuple);
		int y = get<1>(tuple);
		int z = get<2>(tuple);
		int t = get<3>(tuple);

		runner.run("main", x, y, z, t);

		EXPECT_EQ(t, x * y * z);
	}
}

TEST(MulOp, scalarTimesVector)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer[3] y;
	 *   output Integer[3] z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Integer>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(3), "z"),
			Expression::operation(location, makeType<BuiltInType::Integer>(3), OperationKind::multiply,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
														Expression::reference(location, makeType<BuiltInType::Integer>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	int x = 2;
	array<int, 2> y = { 3, -5 };
	array<int, 2> z = { 0, 0 };

	int* yPtr = y.data();
	int* zPtr = z.data();

	runner.run("main", x, yPtr, zPtr);

	for (const auto& tuple : llvm::zip(y, z))
		EXPECT_EQ(get<1>(tuple), x * get<0>(tuple));
}

TEST(MulOp, vectorTimesScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[3] x;
	 *   input Integer y;
	 *   output Integer[3] z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Integer>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(3), "z"),
			Expression::operation(location, makeType<BuiltInType::Integer>(3), OperationKind::multiply,
														Expression::reference(location, makeType<BuiltInType::Integer>(3), "x"),
														Expression::reference(location, makeType<BuiltInType::Integer>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<int, 2> x = { 3, -5 };
	int y = 2;
	array<int, 2> z = { 0, 0 };

	int* xPtr = x.data();
	int* zPtr = z.data();

	runner.run("main", xPtr, y, zPtr);

	for (const auto& tuple : llvm::zip(x, z))
		EXPECT_EQ(get<1>(tuple), get<0>(tuple) * y);
}

/*
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
*/