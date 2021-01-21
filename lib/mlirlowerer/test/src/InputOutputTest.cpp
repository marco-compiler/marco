#include <gtest/gtest.h>
#include <mlir/IR/Dialect.h>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/mlirlowerer/Runner.hpp>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
using namespace std;

TEST(Input, integerScalar)	 // NOLINT
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

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	int x = 57;
	int y = 0;

	Runner runner(&context, module);
	runner.run("main", x, y);

	EXPECT_EQ(y, 57);
}

TEST(Input, floatScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   output Real y;
	 *   algorithm
	 *     y := x;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Float>(), "y"),
			Expression::reference(location, makeType<BuiltInType::Float>(), "x"));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	float x = 57;
	float y = 0;

	Runner runner(&context, module);
	runner.run("main", x, y);

	EXPECT_EQ(y, 57.0);
}

TEST(Input, integerArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[2] x;
	 *   output Integer y;
	 *   algorithm
	 *     y := x[0] + x[1];
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::add,
														Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
																									Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
														Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
																									Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 1))));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	array<int, 2> x = { 23, 57 };
	int* xPtr = x.data();
	int y = 0;

	Runner runner(&context, module);
	runner.run("main", xPtr, y);

	EXPECT_EQ(y, 80);
}

TEST(Input, floatArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Real[2] x;
	 *   output Real y;
	 *   algorithm
	 *     y := x[0] + x[1];
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Float>(), "y"),
			Expression::operation(location, makeType<BuiltInType::Float>(), OperationKind::add,
														Expression::operation(location, makeType<BuiltInType::Float>(), OperationKind::subscription,
																									Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
														Expression::operation(location, makeType<BuiltInType::Float>(), OperationKind::subscription,
																									Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 1))));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	array<float, 2> x = { 23.0, 57.0 };
	float* xPtr = x.data();
	float y = 0;

	Runner runner(&context, module);
	runner.run("main", xPtr, y);

	EXPECT_FLOAT_EQ(y, 80);
}

TEST(Output, integerArray)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer[2] x;
	 *   algorithm
	 *     y[0] := 23;
	 *     y[1] := 57;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment0 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
														Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23));

	Statement assignment1 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
														Expression::constant(location, makeType<BuiltInType::Integer>(), 1)),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57));

	ClassContainer cls(Function(location, "main", true, xMember, Algorithm(location, { assignment0, assignment1 })));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	array<int, 2> x = { 0, 0 };
	int* xPtr = x.data();

	Runner runner(&context, module);
	runner.run("main", xPtr);

	EXPECT_EQ(x[0], 23);
	EXPECT_EQ(x[1], 57);
}
