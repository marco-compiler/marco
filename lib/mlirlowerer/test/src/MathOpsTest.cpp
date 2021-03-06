#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/mlirlowerer/CRunnerUtils.h>
#include <modelica/mlirlowerer/MlirLowerer.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
using namespace std;

TEST(MathOps, sumOfIntegerScalars)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
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

	array<int, 2> x = { 23, 57 };
	array<int, 2> y = { 57, -23 };
	array<int, 2> z = { 0, 0 };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x + y);
	}
}

TEST(MathOps, sumOfIntegerStaticArrays)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::add,
														Expression::reference(location, makeType<int>(3), "x"),
														Expression::reference(location, makeType<int>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 3 });

	if (failed(runner.run("main", xPtr, yPtr, Runner::result(zPtr))))
		FAIL();

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
		EXPECT_EQ(z, x + y);
}

TEST(MathOps, sumOfIntegerDynamicArrays)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[:] x;
	 *   input Integer[:] y;
	 *   output Integer[:] z;
	 *
	 *   algorithm
	 *     z := x + y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "z"),
			Expression::operation(location, makeType<int>(-1), OperationKind::add,
														Expression::reference(location, makeType<int>(-1), "x"),
														Expression::reference(location, makeType<int>(-1), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 3 });

	if (failed(runner.run("main", xPtr, yPtr, Runner::result(zPtr))))
		FAIL();

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
		EXPECT_EQ(z, x + y);
}

TEST(MathOps, sumOfFloatScalars)	 // NOLINT
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

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::add,
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

	array<float, 2> x = { 23.2f, 57.5f };
	array<float, 2> y = { 57.3f, -23.7f };
	array<float, 2> z = { 0.0f, 0.0f };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_FLOAT_EQ(z, x + y);
	}
}

TEST(MathOps, sumOfFloatStaticArrays)	 // NOLINT
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

	Member xMember(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(3), "z"),
			Expression::operation(location, makeType<float>(3), OperationKind::add,
														Expression::reference(location, makeType<float>(3), "x"),
														Expression::reference(location, makeType<float>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<float, 3> x = { 10.1f, 23.3f, 57.8f };
	array<float, 3> y = { 10.2f, 57.3f, -23.5f };
	array<float, 3> z = { 0.0f, 0.0f, 0.0f };

	ArrayDescriptor<float, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<float, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<float, 1> zPtr(z.data(), { 3 });

	if (failed(runner.run("main", xPtr, yPtr, Runner::result(zPtr))))
		FAIL();

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
		EXPECT_FLOAT_EQ(z, x + y);
}

TEST(MathOps, sumOfFloatDynamicArrays)	 // NOLINT
{
	/**
	 * function main
	 *   input Real[:] x;
	 *   input Real[:] y;
	 *   output Real[:] z;
	 *
	 *   algorithm
	 *     z := x + y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(-1), "z"),
			Expression::operation(location, makeType<float>(-1), OperationKind::add,
														Expression::reference(location, makeType<float>(-1), "x"),
														Expression::reference(location, makeType<float>(-1), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<float, 3> x = { 10.1f, 23.3f, 57.8f };
	array<float, 3> y = { 10.2f, 57.3f, -23.5f };
	array<float, 3> z = { 0.0f, 0.0f, 0.0f };

	ArrayDescriptor<float, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<float, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<float, 1> zPtr(z.data(), { 3 });

	if (failed(runner.run("main", xPtr, yPtr, Runner::result(zPtr))))
		FAIL();

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
		EXPECT_FLOAT_EQ(z, x + y);
}

TEST(MathOps, sumIntegerScalarAndFloatScalar)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::add,
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

	array<int, 3> x = { 2, -3, -3 };
	array<float, 3> y = { -3.5f, 5.2f, -2.0f };
	array<float, 3> z = { 0.0f, 0.0f, 0.0f };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_FLOAT_EQ(z, x + y);
	}
}

TEST(MathOps, sumIntegerArrayAndFloatArray)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(3), "z"),
			Expression::operation(location, makeType<float>(3), OperationKind::add,
														Expression::reference(location, makeType<int>(3), "x"),
														Expression::reference(location, makeType<float>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 2, -3, -3 };
	array<float, 3> y = { -3.5f, 5.2f, -2.0f };
	array<float, 3> z = { 0.0f, 0.0f, 0.0f };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<float, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<float, 1> zPtr(z.data(), { 3 });

	if (failed(runner.run("main", xPtr, yPtr, Runner::result(zPtr))))
		FAIL();

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
		EXPECT_FLOAT_EQ(z, x + y);
}

TEST(MathOps, sumMultipleIntegerScalars)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member tMember(location, "t", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "t"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y"),
														Expression::reference(location, makeType<int>(), "z")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember, tMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 4, -7, -15 };
	array<int, 3> t = { 0, 0, 0 };

	for (const auto& [x, y, z, t] : llvm::zip(x, y, z, t))
	{
		if (failed(runner.run("main", x, y, z, Runner::result(t))))
			FAIL();

		EXPECT_EQ(t, x + y + z);
	}
}

TEST(MathOps, subOfIntegerScalars)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::subtract,
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

	array<int, 2> x = { 23, 57 };
	array<int, 2> y = { 57, -23 };
	array<int, 2> z = { 0, 0 };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x - y);
	}
}

TEST(MathOps, subOfIntegerStaticArrays)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::subtract,
														Expression::reference(location, makeType<int>(3), "x"),
														Expression::reference(location, makeType<int>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	module.dump();

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 3 });

	if (failed(runner.run("main", xPtr, yPtr, Runner::result(zPtr))))
		FAIL();

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
		EXPECT_EQ(z, x - y);
}

TEST(MathOps, subOfIntegerDynamicArrays)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[:] x;
	 *   input Integer[:] y;
	 *   output Integer[:] z;
	 *
	 *   algorithm
	 *     z := x - y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "z"),
			Expression::operation(location, makeType<int>(-1), OperationKind::subtract,
														Expression::reference(location, makeType<int>(-1), "x"),
														Expression::reference(location, makeType<int>(-1), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	module.dump();

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 3 });

	if (failed(runner.run("main", xPtr, yPtr, Runner::result(zPtr))))
		FAIL();

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
		EXPECT_EQ(z, x - y);
}

TEST(MathOps, subOfFloatScalars)	 // NOLINT
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

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::subtract,
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

	array<float, 2> x = { 23.2f, 57.5f };
	array<float, 2> y = { 57.3f, -23.7f };
	array<float, 2> z = { 0.0f, 0.0f };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_FLOAT_EQ(z, x - y);
	}
}

TEST(MathOps, subOfFloatStaticArrays)	 // NOLINT
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

	Member xMember(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(3), "z"),
			Expression::operation(location, makeType<float>(3), OperationKind::subtract,
														Expression::reference(location, makeType<float>(3), "x"),
														Expression::reference(location, makeType<float>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<float, 3> x = { 10.1f, 23.3f, 57.8f };
	array<float, 3> y = { 10.2f, 57.3f, -23.5f };
	array<float, 3> z = { 0.0f, 0.0f, 0.0f };

	ArrayDescriptor<float, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<float, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<float, 1> zPtr(z.data(), { 3 });

	if (failed(runner.run("main", xPtr, yPtr, Runner::result(zPtr))))
		FAIL();

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
		EXPECT_FLOAT_EQ(z, x - y);
}

TEST(MathOps, subOfFloatDynamicArrays)	 // NOLINT
{
	/**
	 * function main
	 *   input Real[-1] x;
	 *   input Real[-1] y;
	 *   output Real[-1] z;
	 *
	 *   algorithm
	 *     z := x - y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(-1), "z"),
			Expression::operation(location, makeType<float>(-1), OperationKind::subtract,
														Expression::reference(location, makeType<float>(-1), "x"),
														Expression::reference(location, makeType<float>(-1), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<float, 3> x = { 10.1f, 23.3f, 57.8f };
	array<float, 3> y = { 10.2f, 57.3f, -23.5f };
	array<float, 3> z = { 0.0f, 0.0f, 0.0f };

	ArrayDescriptor<float, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<float, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<float, 1> zPtr(z.data(), { 3 });

	if (failed(runner.run("main", xPtr, yPtr, Runner::result(zPtr))))
		FAIL();

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
		EXPECT_FLOAT_EQ(z, x - y);
}

TEST(MathOps, subIntegerScalarAndFloatScalar)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::subtract,
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

	array<int, 3> x = { 2, -3, -3 };
	array<float, 3> y = { -3.5f, 5.2f, -2.0f };
	array<float, 3> z = { 0.0f, 0.0f, 0.0f };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_FLOAT_EQ(z, x - y);
	}
}

TEST(MathOps, subIntegerArrayAndFloatArray)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(3), "z"),
			Expression::operation(location, makeType<float>(3), OperationKind::subtract,
														Expression::reference(location, makeType<int>(3), "x"),
														Expression::reference(location, makeType<float>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 2, -3, -3 };
	array<float, 3> y = { -3.5f, 5.2f, -2.0f };
	array<float, 3> z = { 0.0f, 0.0f, 0.0f };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<float, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<float, 1> zPtr(z.data(), { 3 });

	if (failed(runner.run("main", xPtr, yPtr, Runner::result(zPtr))))
		FAIL();

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
		EXPECT_FLOAT_EQ(z, x - y);
}

TEST(MathOps, subMultipleIntegerScalars)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member tMember(location, "t", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "t"),
			Expression::operation(location, makeType<int>(), OperationKind::subtract,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y"),
														Expression::reference(location, makeType<int>(), "z")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember, tMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 4, -7, -15 };
	array<int, 3> t = { 0, 0, 0 };

	for (const auto& [x, y, z, t] : llvm::zip(x, y, z, t))
	{
		if (failed(runner.run("main", x, y, z, Runner::result(t))))
			FAIL();

		EXPECT_EQ(t, x - y - z);
	}
}

TEST(MathOps, mulOfIntegerScalars)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::multiply,
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

	array<int, 2> x = { 2, 5 };
	array<int, 2> y = { 3, -3 };
	array<int, 2> z = { 0, 0 };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x * y);
	}
}

TEST(MathOps, mulOfFloatScalars)	 // NOLINT
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

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::multiply,
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

	array<float, 2> x = { 2.3f, 5.7f };
	array<float, 2> y = { 23.57f, -23.57f };
	array<float, 2> z = { 0.0f, 0.0f };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_FLOAT_EQ(z, x * y);
	}
}

TEST(MathOps, mulIntegerScalarAndFloatScalar)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::multiply,
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

	array<int, 3> x = { 2, -3, -3 };
	array<float, 3> y = { -3.5f, 5.2f, -2.0f };
	array<float, 3> z = { 0.0f, 0.0f, 0.0f };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_FLOAT_EQ(z, x * y);
	}
}

TEST(MathOps, mulMultipleIntegerScalars)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member tMember(location, "t", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "t"),
			Expression::operation(location, makeType<int>(), OperationKind::multiply,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y"),
														Expression::reference(location, makeType<int>(), "z")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember, tMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 4, -7, -15 };
	array<int, 3> t = { 0, 0, 0 };

	for (const auto& [x, y, z, t] : llvm::zip(x, y, z, t))
	{
		if (failed(runner.run("main", x, y, z, Runner::result(t))))
			FAIL();

		EXPECT_EQ(t, x * y * z);
	}
}

TEST(MathOps, mulIntegerScalarAndIntegerStaticArray)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::multiply,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	int x = 2;
	array<int, 3> y = { 3, -5, 0 };

	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(nullptr, { 3 });

	if (failed(runner.run("main", x, yPtr, Runner::result(zPtr))))
		FAIL();

	for (const auto& [y, z] : llvm::zip(yPtr, zPtr))
		EXPECT_EQ(z, x * y);
}

TEST(MathOps, mulIntegerScalarAndIntegerDynamicArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer[-1] y;
	 *   output Integer[-1] z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "z"),
			Expression::operation(location, makeType<int>(-1), OperationKind::multiply,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(-1), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	int x = 2;
	array<int, 3> y = { 3, -5, 0 };

	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(nullptr, { 3 });

	if (failed(runner.run("main", x, yPtr, Runner::result(zPtr))))
		FAIL();

	for (const auto& [y, z] : llvm::zip(yPtr, zPtr))
		EXPECT_EQ(z, x * y);
}

TEST(MathOps, mulIntegerStaticArrayAndIntegerScalar)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::multiply,
														Expression::reference(location, makeType<int>(3), "x"),
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

	array<int, 3> x = { 3, -5, 0 };
	int y = 2;

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(nullptr, { 3 });

	if (failed(runner.run("main", xPtr, y, Runner::result(zPtr))))
		FAIL();

	for (const auto& [x, z] : llvm::zip(xPtr, zPtr))
		EXPECT_EQ(z, x * y);
}

TEST(MathOps, mulIntegerDynamicArrayAndIntegerScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[-1] x;
	 *   input Integer y;
	 *   output Integer[-1] z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "z"),
			Expression::operation(location, makeType<int>(-1), OperationKind::multiply,
														Expression::reference(location, makeType<int>(-1), "x"),
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

	array<int, 3> x = { 3, -5, 0 };
	int y = 2;

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(nullptr, { 3 });

	if (failed(runner.run("main", xPtr, y, Runner::result(zPtr))))
		FAIL();

	for (const auto& [x, z] : llvm::zip(xPtr, zPtr))
		EXPECT_EQ(z, x * y);
}

TEST(MathOps, mulCrossProductIntegerStaticArrays)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[3] x;
	 *   input Integer[3] y;
	 *   output Integer z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::multiply,
														Expression::reference(location, makeType<int>(3), "x"),
														Expression::reference(location, makeType<int>(3), "y")));

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

	array<int, 3> x = { 3, 5, 2 };
	array<int, 3> y = { 7, -2, 3 };
	int z = 0;

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });

	if (failed(runner.run("main", xPtr, yPtr, Runner::result(z))))
		FAIL();

	EXPECT_EQ(z, 17);
}

TEST(MathOps, divOfIntegerScalars)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Integer z;
	 *
	 *   algorithm
	 *     z := x / y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::divide,
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

	array<int, 2> x = { 6, 10 };
	array<int, 2> y = { 3, -5 };
	array<int, 2> z = { 0, 0 };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x / y);
	}
}

TEST(MathOps, divOfFloatScalars)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Real z;
	 *
	 *   algorithm
	 *     z := x / y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::divide,
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

	array<float, 2> x = { 10.8f, 10.0f };
	array<float, 2> y = { 3.6f, -3.2f };
	array<float, 2> z = { 0.0f, 0.0f };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_FLOAT_EQ(z, x / y);
	}
}

TEST(MathOps, divMultipleIntegerScalars)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   input Integer z;
	 *   output Integer t;
	 *
	 *   algorithm
	 *     t := x / y / z;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member tMember(location, "t", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "t"),
			Expression::operation(location, makeType<int>(), OperationKind::divide,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y"),
														Expression::reference(location, makeType<int>(), "z")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember, tMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 120, 50, 0 };
	array<int, 3> y = { 2, 5, 5 };
	array<int, 3> z = { -3, 2, 2 };
	array<int, 3> t = { 0, 0, 0 };

	for (const auto& [x, y, z, t] : llvm::zip(x, y, z, t))
	{
		if (failed(runner.run("main", x, y, z, Runner::result(t))))
			FAIL();

		EXPECT_EQ(t, x / y / z);
	}
}

TEST(MathOps, divIntegerScalarAndFloatScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Real z;
	 *
	 *   algorithm
	 *     z := x / y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::divide,
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

	array<int, 3> x = { 23, 10, -3 };
	array<float, 3> y = { -3.5f, 3.2f, -2.0f };
	array<float, 3> z = { 0.0f, 0.0f, 0.0f };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_FLOAT_EQ(z, x / y);
	}
}

TEST(MathOps, divIntegerStaticArrayAndIntegerScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[3] x;
	 *   input Integer y;
	 *   output Integer[3] z;
	 *
	 *   algorithm
	 *     z := x / y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::divide,
														Expression::reference(location, makeType<int>(3), "x"),
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

	array<int, 3> x = { 10, -5, 0 };
	int y = 2;

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(nullptr, { 3 });

	if (failed(runner.run("main", xPtr, y, Runner::result(zPtr))))
		FAIL();

	for (const auto& [x, z] : llvm::zip(xPtr, zPtr))
		EXPECT_EQ(z, x / y);
}

TEST(MathOps, divIntegerDynamicArrayAndIntegerScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[-1] x;
	 *   input Integer y;
	 *   output Integer[-1] z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "z"),
			Expression::operation(location, makeType<int>(-1), OperationKind::divide,
														Expression::reference(location, makeType<int>(-1), "x"),
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

	array<int, 3> x = { 3, -5, 0 };
	int y = 2;

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(nullptr, { 3 });

	if (failed(runner.run("main", xPtr, y, Runner::result(zPtr))))
		FAIL();

	for (const auto& [x, z] : llvm::zip(xPtr, zPtr))
		EXPECT_EQ(z, x / y);
}

/*

TEST(Pow, integerRaisedToInteger)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	Expression expression = Expression::operation(
			location,
			makeType<int>(),
			OperationKind::powerOf,
			Expression::constant(location, makeType<int>(), 2),
			Expression::constant(location, makeType<int>(), 3));

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
			makeType<int>(),
			OperationKind::powerOf,
			Expression::constant(location, makeType<int>(), 4),
			Expression::constant(location, makeType<float>(), 0.5));

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
			makeType<float>(),
			OperationKind::powerOf,
			Expression::constant(location, makeType<float>(), 2.5),
			Expression::constant(location, makeType<int>(), 2));

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
			makeType<float>(),
			OperationKind::powerOf,
			Expression::constant(location, makeType<float>(), 1.5625),
			Expression::constant(location, makeType<int>(), 0.5));

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