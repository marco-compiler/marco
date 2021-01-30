#include <gtest/gtest.h>
#include <mlir/IR/Dialect.h>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/mlirlowerer/Runner.hpp>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
using namespace std;

TEST(Assignment, constant)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer x;
	 *
	 *   algorithm
	 *     x := 57;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57));

	ClassContainer cls(Function(location, "main", true, xMember, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int x = 0;

	Runner runner(&context, module);
	runner.run("main", x);

	EXPECT_EQ(x, 57);
}

TEST(Assignment, variableCopy)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := x;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
			Expression::reference(location, makeType<BuiltInType::Integer>(), "x"));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int x = 57;
	int y = 0;

	Runner runner(&context, module);
	runner.run("main", x, y);

	EXPECT_EQ(y, x);
}

TEST(Assignment, arraySliceAssignment)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[2] x;
	 *   input Integer[2] y;
	 *   input Integer[2] z;
	 *   output Integer[3,2] t;
	 *
	 *   algorithm
	 *     t[1] := x;
	 *     t[2] := y;
	 *     t[3] := z;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Integer>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member tMember(location, "t", makeType<BuiltInType::Integer>(3, 2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment1 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<BuiltInType::Integer>(2), OperationKind::subscription,
														Expression::reference(location, makeType<BuiltInType::Integer>(3, 2), "t"),
														Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
			Expression::reference(location, makeType<BuiltInType::Integer>(2), "x"));

	Statement assignment2 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<BuiltInType::Integer>(2), OperationKind::subscription,
														Expression::reference(location, makeType<BuiltInType::Integer>(3, 2), "t"),
														Expression::constant(location, makeType<BuiltInType::Integer>(), 1)),
			Expression::reference(location, makeType<BuiltInType::Integer>(2), "y"));

	Statement assignment3 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<BuiltInType::Integer>(2), OperationKind::subscription,
														Expression::reference(location, makeType<BuiltInType::Integer>(3, 2), "t"),
														Expression::constant(location, makeType<BuiltInType::Integer>(), 2)),
			Expression::reference(location, makeType<BuiltInType::Integer>(2), "z"));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember, tMember },
															Algorithm(location, { assignment1, assignment2, assignment3 })));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	array<int, 2> x = { 0, 1 };
	array<int, 2> y = { 2, 3 };
	array<int, 2> z = { 4, 5 };
	array<int, 6> t = { 0, 0, 0, 0, 0, 0 };

	int* xPtr = x.data();
	int* yPtr = y.data();
	int* zPtr = z.data();
	int* tPtr = t.data();

	Runner runner(&context, module);
	runner.run("main", xPtr, yPtr, zPtr, tPtr);

	EXPECT_EQ(t[0], x[0]);
	EXPECT_EQ(t[1], x[1]);
	EXPECT_EQ(t[2], y[0]);
	EXPECT_EQ(t[3], y[1]);
	EXPECT_EQ(t[4], z[0]);
	EXPECT_EQ(t[5], z[1]);
}

TEST(Assignment, arrayCopy)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[2] x;
	 *   output Integer[2] y;
	 *
	 *   algorithm
	 *     y := x;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(2), "y"),
			Expression::reference(location, makeType<BuiltInType::Integer>(2), "x"));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	array<int, 2> x = { 23, 57 };
	int* xPtr = x.data();

	array<int, 2> y = { 0, 0 };
	int* yPtr = y.data();

	Runner runner(&context, module);
	runner.run("main", xPtr, yPtr);

	EXPECT_EQ(y[0], x[0]);
	EXPECT_EQ(y[1], x[1]);
}

TEST(Assignment, internalArrayElement)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *
	 *   protected
	 *     Integer[2] z;
	 *
	 *   algorithm
	 *     z[0] := x * 2;
	 *     z[1] := z[0] + 1;
	 *     y := z[1];
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Member zMember(location, "z", makeType<BuiltInType::Integer>(2), TypePrefix(ParameterQualifier::none, IOQualifier::none));

	Algorithm algorithm(
			location,
			{
					AssignmentStatement(
							location,
							Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
																		Expression::reference(location, makeType<BuiltInType::Integer>(2), "z"),
																		Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
							Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::multiply,
																		Expression::reference(location, makeType<BuiltInType::Integer>(2), "x"),
																		Expression::constant(location, makeType<BuiltInType::Integer>(), 2))),
					AssignmentStatement(
							location,
							Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
																		Expression::reference(location, makeType<BuiltInType::Integer>(2), "z"),
																		Expression::constant(location, makeType<BuiltInType::Integer>(), 1)),
							Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::add,
																		Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
																													Expression::reference(location, makeType<BuiltInType::Integer>(2), "z"),
																													Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
																		Expression::constant(location, makeType<BuiltInType::Integer>(), 1))),
					AssignmentStatement(
							location,
							Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
							Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
																		Expression::reference(location, makeType<BuiltInType::Integer>(2), "z"),
																		Expression::constant(location, makeType<BuiltInType::Integer>(), 1)))
			});

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															algorithm));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	int x = 57;
	int y = 0;

	Runner runner(&context, module);
	runner.run("main", x, y);

	EXPECT_EQ(y, x * 2 + 1);
}
