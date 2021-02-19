#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/mlirlowerer/MlirLowerer.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
using namespace std;

TEST(EqOp, integers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x == y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::equal,
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

	array<int, 2> xData = { 57, 57 };
	array<int, 2> yData = { 57, 23 };
	array<bool, 2> zData = { false, true };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		int y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x == y);
	}
}

TEST(EqOp, floats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x == y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::equal,
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

	array<float, 2> xData = { 57.0f, 57.0f };
	array<float, 2> yData = { 57.0f, 23.0f };
	array<bool, 2> zData = { false, true };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		float x = get<0>(tuple);
		float y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x == y);
	}
}

TEST(EqOp, integerCastedToFloat)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x == y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::equal,
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

	array<int, 2> xData = { 57, 57 };
	array<float, 2> yData = { 57.0f, 23.0f };
	array<bool, 2> zData = { false, true };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		float y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x == y);
	}
}

TEST(NotEqOp, integers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x != y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::different,
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

	array<int, 2> xData = { 57, 57 };
	array<int, 2> yData = { 57, 23 };
	array<bool, 2> zData = { true, false };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		int y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x != y);
	}
}

TEST(NotEqOp, floats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x != y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::different,
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

	array<float, 2> xData = { 57.0f, 57.0f };
	array<float, 2> yData = { 57.0f, 23.0f };
	array<bool, 2> zData = { true, false };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		float x = get<0>(tuple);
		float y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x != y);
	}
}

TEST(NotEqOp, integerCastedToFloat)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x != y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::different,
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

	array<int, 2> xData = { 57, 57 };
	array<float, 2> yData = { 57.0f, 23.0f };
	array<bool, 2> zData = { true, false };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		float y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x != y);
	}
}

TEST(GtOp, integers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x > y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::greater,
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

	array<int, 3> xData = { 23, 57, 57 };
	array<int, 3> yData = { 57, 57, 23 };
	array<bool, 3> zData = { true, true, false };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		int y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x > y);
	}
}

TEST(GtOp, floats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x > y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::greater,
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

	array<float, 3> xData = { 23.0f, 57.0f, 57.0f };
	array<float, 3> yData = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> zData = { true, true, false };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		float x = get<0>(tuple);
		float y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x > y);
	}
}

TEST(GtOp, integerCastedToFloat)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x > y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::greater,
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

	array<int, 3> xData = { 23, 57, 57 };
	array<float, 3> yData = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> zData = { true, true, false };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		float y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x > y);
	}
}

TEST(GteOp, integers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x >= y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::greaterEqual,
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

	array<int, 3> xData = { 23, 57, 57 };
	array<int, 3> yData = { 57, 57, 23 };
	array<bool, 3> zData = { true, false, false };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		int y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x >= y);
	}
}

TEST(GteOp, floats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x >= y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::greaterEqual,
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

	array<float, 3> xData = { 23.0f, 57.0f, 57.0f };
	array<float, 3> yData = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> zData = { true, false, false };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		float x = get<0>(tuple);
		float y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x >= y);
	}
}

TEST(GteOp, integerCastedToFloat)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x >= y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::greaterEqual,
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

	array<int, 3> xData = { 23, 57, 57 };
	array<float, 3> yData = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> zData = { true, false, false };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		float y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x >= y);
	}
}

TEST(LtOp, integers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x < y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::less,
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

	array<int, 3> xData = { 23, 57, 57 };
	array<int, 3> yData = { 57, 57, 23 };
	array<bool, 3> zData = { false, true, true };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		int y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x < y);
	}
}

TEST(LtOp, floats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x < y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::less,
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

	array<float, 3> xData = { 23.0f, 57.0f, 57.0f };
	array<float, 3> yData = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> zData = { false, true, true };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		float x = get<0>(tuple);
		float y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x < y);
	}
}

TEST(LtOp, integerCastedToFloat)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x < y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::less,
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

	array<int, 3> xData = { 23, 57, 57 };
	array<float, 3> yData = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> zData = { false, true, true };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		float y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x < y);
	}
}

TEST(LteOp, integers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x <= y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::lessEqual,
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

	array<int, 3> xData = { 23, 57, 57 };
	array<int, 3> yData = { 57, 57, 23 };
	array<bool, 3> zData = { false, false, true };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		int y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x <= y);
	}
}

TEST(LteOp, floats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x <= y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::lessEqual,
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

	array<float, 3> xData = { 23.0f, 57.0f, 57.0f };
	array<float, 3> yData = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> zData = { false, false, true };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		float x = get<0>(tuple);
		float y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x <= y);
	}
}

TEST(LteOp, integerCastedToFloat)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x <= y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::lessEqual,
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

	array<int, 3> xData = { 23, 57, 57 };
	array<float, 3> yData = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> zData = { false, false, true };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		int x = get<0>(tuple);
		float y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x <= y);
	}
}
