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
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);
	int x = 0;
	runner.run("main", x);
	EXPECT_EQ(x, 57);
}

TEST(Assignment, variableCopy)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
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
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);
	int x = 57;
	int y = 0;
	runner.run("main", x, y);
	EXPECT_EQ(y, x);
}

TEST(Assignment, arrayCopy)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[2] x;
	 *   output Integer[2] y;
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
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);

	array<int, 2> x = { 23, 57 };
	int* xPtr = x.data();

	array<int, 2> y = { 0, 0 };
	int* yPtr = y.data();

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
	 *   protected
	 *     Integer[2] z;
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
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);
	int x = 57;
	int y = 0;
	runner.run("main", x, y);
	EXPECT_EQ(y, x * 2 + 1);
}
