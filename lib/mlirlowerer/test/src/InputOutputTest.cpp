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
	 *     y := x[1] + x[2];
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
	 *     y := x[1] + x[2];
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

TEST(Input, integerMatrix)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[2,3] x;
	 *   output Integer y;
	 *   output Integer z;
	 *   algorithm
	 *     y := x[1][1] + x[1][2] + x[1][3];
	 *     z := x[2][1] + x[2][2] + x[2][3];
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(2, 3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Member zMember(location, "z", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement yAssignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::add,
														Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
																									Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 0),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
														Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
																									Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 0),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 1)),
														Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
																									Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 0),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 2))));

	Statement zAssignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::add,
														Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
																									Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 1),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
														Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
																									Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 1),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 1)),
														Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
																									Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 1),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 2))));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember, zMember }, Algorithm(location, { yAssignment, zAssignment })));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	array<int, 6> x = { 1, 2, 3, 4, 5, 6 };
	int* xPtr = x.data();

	struct {
		int y = 0;
		int z = 0;
	} result;

	Runner runner(&context, module);
	runner.run("main", xPtr, result);

	EXPECT_EQ(result.y, 6);
	EXPECT_EQ(result.z, 15);
}

TEST(Output, integerArray)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer[2] x;
	 *   algorithm
	 *     x[1] := 23;
	 *     x[2] := 57;
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

TEST(Output, floatArray)	 // NOLINT
{
	/**
	 * function main
	 *   output Real[2] x;
	 *   algorithm
	 *     x[1] := 23;
	 *     x[2] := 57;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment0 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<BuiltInType::Float>(), OperationKind::subscription,
														Expression::reference(location, makeType<BuiltInType::Float>(), "x"),
														Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
			Expression::constant(location, makeType<BuiltInType::Float>(), 23.0));

	Statement assignment1 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<BuiltInType::Float>(), OperationKind::subscription,
														Expression::reference(location, makeType<BuiltInType::Float>(), "x"),
														Expression::constant(location, makeType<BuiltInType::Integer>(), 1)),
			Expression::constant(location, makeType<BuiltInType::Float>(), 57.0));

	ClassContainer cls(Function(location, "main", true, xMember, Algorithm(location, { assignment0, assignment1 })));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	array<float, 2> x = { 0, 0 };
	float* xPtr = x.data();

	Runner runner(&context, module);
	runner.run("main", xPtr);

	EXPECT_EQ(x[0], 23.0);
	EXPECT_EQ(x[1], 57.0);
}

TEST(Output, integerMatrix)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer[2,3] x;
	 *   algorithm
	 *     x[1][1] := 1;
	 *     x[1][2] := 2;
	 *     x[1][3] := 3;
	 *     x[2][1] := 4;
	 *     x[2][2] := 5;
	 *     x[2][3] := 6;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(2, 3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	llvm::SmallVector<Statement, 3> assignments;

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 3; j++)
		{
			assignments.push_back(AssignmentStatement(
					location,
					Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
																Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
																Expression::constant(location, makeType<BuiltInType::Integer>(), i),
																Expression::constant(location, makeType<BuiltInType::Integer>(), j)),
					Expression::constant(location, makeType<BuiltInType::Integer>(), i * 3 + j + 1)));
		}

	ClassContainer cls(Function(location, "main", true, xMember, Algorithm(location, assignments)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	array<int, 6> x = { 0, 0, 0, 0, 0, 0 };
	int* xPtr = x.data();

	Runner runner(&context, module);
	runner.run("main", xPtr);

	/*
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 3; j++)
			EXPECT_EQ(x[i * 3 + j], i * 3 + j + 1);
	*/

	EXPECT_EQ(x[0], 1);
	EXPECT_EQ(x[1], 2);
	EXPECT_EQ(x[2], 3);
	EXPECT_EQ(x[3], 4);
	EXPECT_EQ(x[4], 5);
	EXPECT_EQ(x[5], 6);
}