#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/mlirlowerer/MlirLowerer.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
using namespace std;

TEST(Comparison, eqIntegers)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::equal,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 2> x = { 57, 57 };
	array<int, 2> y = { 57, 23 };
	array<bool, 2> z = { false, true };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x == y);
	}
}

TEST(Comparison, eqFloats)	 // NOLINT
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

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::equal,
														Expression::reference(location, makeType<float>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<float, 2> x = { 57.0f, 57.0f };
	array<float, 2> y = { 57.0f, 23.0f };
	array<bool, 2> z = { false, true };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x == y);
	}
}

TEST(Comparison, eqIntegerAndFloat)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::equal,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	module.dump();
	llvm::DebugFlag = true;

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	module.dump();
	llvm::DebugFlag = false;

	Runner runner(module);

	array<int, 2> x = { 57, 57 };
	array<float, 2> y = { 57.0f, 23.0f };
	array<bool, 2> z = { false, true };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x == y);
	}
}

TEST(Comparison, notEqIntegers)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::different,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 2> x = { 57, 57 };
	array<int, 2> y = { 57, 23 };
	array<bool, 2> z = { true, false };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x != y);
	}
}

TEST(Comparison, notEqFloats)	 // NOLINT
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

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::different,
														Expression::reference(location, makeType<float>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<float, 2> x = { 57.0f, 57.0f };
	array<float, 2> y = { 57.0f, 23.0f };
	array<bool, 2> z = { true, false };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x != y);
	}
}

TEST(Comparison, notEqIntegerAndFloat)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::different,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 2> x = { 57, 57 };
	array<float, 2> y = { 57.0f, 23.0f };
	array<bool, 2> z = { true, false };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x != y);
	}
}

TEST(Comparison, gtIntegers)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::greater,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 23, 57, 57 };
	array<int, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { true, true, false };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x > y);
	}
}

TEST(Comparison, gtFloats)	 // NOLINT
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

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::greater,
														Expression::reference(location, makeType<float>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<float, 3> x = { 23.0f, 57.0f, 57.0f };
	array<float, 3> y = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> z = { true, true, false };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x > y);
	}
}

TEST(Comparison, gtIntegerAndFloat)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::greater,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 23, 57, 57 };
	array<float, 3> y = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> z = { true, true, false };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x > y);
	}
}

TEST(Comparison, gteIntegers)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::greaterEqual,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 23, 57, 57 };
	array<int, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { true, false, false };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x >= y);
	}
}

TEST(Comparison, gteFloats)	 // NOLINT
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

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::greaterEqual,
														Expression::reference(location, makeType<float>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<float, 3> x = { 23.0f, 57.0f, 57.0f };
	array<float, 3> y = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> z = { true, false, false };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x >= y);
	}
}

TEST(Comparison, gteIntegerAndFloat)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::greaterEqual,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 23, 57, 57 };
	array<float, 3> y = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> z = { true, false, false };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x >= y);
	}
}

TEST(Comparison, ltIntegers)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::less,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 23, 57, 57 };
	array<int, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { false, true, true };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x < y);
	}
}

TEST(Comparison, ltFloats)	 // NOLINT
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

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::less,
														Expression::reference(location, makeType<float>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<float, 3> x = { 23.0f, 57.0f, 57.0f };
	array<float, 3> y = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> z = { false, true, true };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x < y);
	}
}

TEST(Comparison, ltIntegerAndFloat)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::less,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 23, 57, 57 };
	array<float, 3> y = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> z = { false, true, true };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x < y);
	}
}

TEST(Comparison, lteIntegers)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::lessEqual,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 23, 57, 57 };
	array<int, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { false, false, true };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x <= y);
	}
}

TEST(Comparison, lteFloats)	 // NOLINT
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

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::lessEqual,
														Expression::reference(location, makeType<float>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<float, 3> x = { 23.0f, 57.0f, 57.0f };
	array<float, 3> y = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> z = { false, false, true };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x <= y);
	}
}

TEST(Comparison, lteIntegerAndFloat)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::lessEqual,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 23, 57, 57 };
	array<float, 3> y = { 57.0f, 57.0f, 23.0f };
	array<bool, 3> z = { false, false, true };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x <= y);
	}
}
