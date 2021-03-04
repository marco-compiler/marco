#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/mlirlowerer/CRunnerUtils.h>
#include <modelica/mlirlowerer/MlirLowerer.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
using namespace std;

TEST(Logic, negateScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Boolean x;
	 *   output Boolean y;
	 *
	 *   algorithm
	 *     y := not x;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "y"),
			Expression::operation(location, makeType<bool>(), OperationKind::negate,
														Expression::reference(location, makeType<bool>(), "x")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
			FAIL();

	Runner runner(module);

	array<bool, 2> x = { true, false };
	array<bool, 2> y = { true, false };

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		if (failed(runner.run("main", x, Runner::result(y))))
			FAIL();

		EXPECT_EQ(y, !x);
	}
}

TEST(Logic, negateVector)	 // NOLINT
{
	/**
	 * function main
	 *   input Boolean[2] x;
	 *   output Boolean[2] y;
	 *
	 *   algorithm
	 *     y := not x;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<bool>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<bool>(2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(2), "y"),
			Expression::operation(location, makeType<bool>(2), OperationKind::negate,
														Expression::reference(location, makeType<bool>(2), "x")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<bool, 2> x = { true, false };
	array<bool, 2> y = { true, false };

	ArrayDescriptor<bool, 1> xPtr(x.data(), { 2 });
	ArrayDescriptor<bool, 1> yPtr(y.data(), { 2 });

	if (failed(runner.run("main", xPtr, Runner::result(yPtr))))
		FAIL();

	for (const auto& [x, y] : llvm::zip(xPtr, yPtr))
		EXPECT_EQ(y, !x);
}

TEST(Logic, andScalars)	 // NOLINT
{
	/**
	 * function main
	 *   input Boolean x;
	 *   input Boolean y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x and y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::land,
														Expression::reference(location, makeType<bool>(), "x"),
														Expression::reference(location, makeType<bool>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<bool, 4> x = { false, false, true, true };
	array<bool, 4> y = { false, true, false, true };
	array<bool, 4> z = { true, true, true, false };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x && y);
	}
}

TEST(Logic, orScalars)	 // NOLINT
{
	/**
	 * function main
	 *   input Boolean x;
	 *   input Boolean y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x or y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::lor,
														Expression::reference(location, makeType<bool>(), "x"),
														Expression::reference(location, makeType<bool>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<bool, 4> x = { false, false, true, true };
	array<bool, 4> y = { false, true, false, true };
	array<bool, 4> z = { true, false, false, false };

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		if (failed(runner.run("main", x, y, Runner::result(z))))
			FAIL();

		EXPECT_EQ(z, x || y);
	}
}
