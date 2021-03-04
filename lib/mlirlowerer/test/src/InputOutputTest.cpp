#include <gtest/gtest.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <mlir/IR/Dialect.h>
#include <modelica/mlirlowerer/CRunnerUtils.h>
#include <modelica/mlirlowerer/MlirLowerer.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
using namespace std;

TEST(Input, booleanScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Boolean x;
	 *   output Boolean y;
	 *
	 *   algorithm
	 *     y := x;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "y"),
			Expression::reference(location, makeType<bool>(), "x"));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	bool x = true;
	bool y = false;

	if (failed(runner.run("main", x, Runner::result(y))))
		FAIL();

	EXPECT_EQ(y, x);
}

TEST(Input, integerScalar)	 // NOLINT
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::reference(location, makeType<int>(), "x"));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	int x = 57;
	int y = 0;

	if (failed(runner.run("main", x, Runner::result(y))))
		FAIL();

	EXPECT_EQ(y, x);
}

TEST(Input, floatScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   output Real y;
	 *
	 *   algorithm
	 *     y := x;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "y"),
			Expression::reference(location, makeType<float>(), "x"));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	float x = 57;
	float y = 0;

	if (failed(runner.run("main", x, Runner::result(y))))
		FAIL();

	EXPECT_FLOAT_EQ(y, x);
}

TEST(Input, integerArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x[2];
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := x[1] + x[2];
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														Expression::operation(location, makeType<int>(), OperationKind::subscription,
																									Expression::reference(location, makeType<int>(), "x"),
																									Expression::constant(location, makeType<int>(), 0)),
														Expression::operation(location, makeType<int>(), OperationKind::subscription,
																									Expression::reference(location, makeType<int>(), "x"),
																									Expression::constant(location, makeType<int>(), 1))));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	module.dump();

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	module.dump();

	Runner runner(module);
	
	array<int, 2> x = { 23, 57 };
	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });

	int y = 0;

	if (failed(runner.run("main", xPtr, Runner::result(y))))
		FAIL();

	EXPECT_EQ(y, x[0] + x[1]);
}

TEST(Input, integerArrayUnknownSize)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x[:];
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := x[1] + x[2];
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														Expression::operation(location, makeType<int>(), OperationKind::subscription,
																									Expression::reference(location, makeType<int>(), "x"),
																									Expression::constant(location, makeType<int>(), 0)),
														Expression::operation(location, makeType<int>(), OperationKind::subscription,
																									Expression::reference(location, makeType<int>(), "x"),
																									Expression::constant(location, makeType<int>(), 1))));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 5> x = { 23, 57, 10, -23, -10 };
	ArrayDescriptor<int, 1> xPtr(x.data(), { 5 });

	int y = 0;

	if (failed(runner.run("main", xPtr, Runner::result(y))))
		FAIL();

	EXPECT_EQ(y, x[0] + x[1]);
}

TEST(Input, floatArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x[2];
	 *   output Real y;
	 *
	 *   algorithm
	 *     y := x[1] + x[2];
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "y"),
			Expression::operation(location, makeType<float>(), OperationKind::add,
														Expression::operation(location, makeType<float>(), OperationKind::subscription,
																									Expression::reference(location, makeType<float>(), "x"),
																									Expression::constant(location, makeType<int>(), 0)),
														Expression::operation(location, makeType<float>(), OperationKind::subscription,
																									Expression::reference(location, makeType<float>(), "x"),
																									Expression::constant(location, makeType<int>(), 1))));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<float, 2> x = { 23.0, 57.0 };
	ArrayDescriptor<float, 1> xPtr(x.data(), { 3 });

	float y = 0;

	if (failed(runner.run("main", xPtr, Runner::result(y))))
		FAIL();

	EXPECT_FLOAT_EQ(y, 80);
}

TEST(Input, floatArrayUnknownSize)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x[:];
	 *   output Real y;
	 *
	 *   algorithm
	 *     y := x[1] + x[2];
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "y"),
			Expression::operation(location, makeType<float>(), OperationKind::add,
														Expression::operation(location, makeType<float>(), OperationKind::subscription,
																									Expression::reference(location, makeType<float>(), "x"),
																									Expression::constant(location, makeType<int>(), 0)),
														Expression::operation(location, makeType<float>(), OperationKind::subscription,
																									Expression::reference(location, makeType<float>(), "x"),
																									Expression::constant(location, makeType<int>(), 1))));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<float, 2> x = { 23.0, 57.0 };
	ArrayDescriptor<float, 1> xPtr(x.data(), { 3 });

	float y = 0;

	if (failed(runner.run("main", xPtr, Runner::result(y))))
		FAIL();

	EXPECT_FLOAT_EQ(y, 80);
}

TEST(Input, integerMatrix)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x[2,3];
	 *   output Integer y;
	 *   output Integer z;
	 *
	 *   algorithm
	 *     y := x[1][1] + x[1][2] + x[1][3];
	 *     z := x[2][1] + x[2][2] + x[2][3];
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(2, 3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement yAssignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														Expression::operation(location, makeType<int>(), OperationKind::subscription,
																									Expression::reference(location, makeType<int>(), "x"),
																									Expression::constant(location, makeType<int>(), 0),
																									Expression::constant(location, makeType<int>(), 0)),
														Expression::operation(location, makeType<int>(), OperationKind::subscription,
																									Expression::reference(location, makeType<int>(), "x"),
																									Expression::constant(location, makeType<int>(), 0),
																									Expression::constant(location, makeType<int>(), 1)),
														Expression::operation(location, makeType<int>(), OperationKind::subscription,
																									Expression::reference(location, makeType<int>(), "x"),
																									Expression::constant(location, makeType<int>(), 0),
																									Expression::constant(location, makeType<int>(), 2))));

	Statement zAssignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														Expression::operation(location, makeType<int>(), OperationKind::subscription,
																									Expression::reference(location, makeType<int>(), "x"),
																									Expression::constant(location, makeType<int>(), 1),
																									Expression::constant(location, makeType<int>(), 0)),
														Expression::operation(location, makeType<int>(), OperationKind::subscription,
																									Expression::reference(location, makeType<int>(), "x"),
																									Expression::constant(location, makeType<int>(), 1),
																									Expression::constant(location, makeType<int>(), 1)),
														Expression::operation(location, makeType<int>(), OperationKind::subscription,
																									Expression::reference(location, makeType<int>(), "x"),
																									Expression::constant(location, makeType<int>(), 1),
																									Expression::constant(location, makeType<int>(), 2))));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember, zMember }, Algorithm(location, { yAssignment, zAssignment })));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 6> x = { 1, 2, 3, 4, 5, 6 };
	ArrayDescriptor<int, 2> xPtr(x.data(), { 2, 3 });

	struct {
		int y = 0;
		int z = 0;
	} result;

	if (failed(runner.run("main", xPtr, Runner::result(result))))
		FAIL();

	EXPECT_EQ(result.y, 6);
	EXPECT_EQ(result.z, 15);
}

TEST(Output, integerArray)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer x[2];
	 *
	 *   algorithm
	 *     x[1] := 23;
	 *     x[2] := 57;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment0 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<int>(), OperationKind::subscription,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::constant(location, makeType<int>(), 0)),
			Expression::constant(location, makeType<int>(), 23));

	Statement assignment1 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<int>(), OperationKind::subscription,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::constant(location, makeType<int>(), 1)),
			Expression::constant(location, makeType<int>(), 57));

	ClassContainer cls(Function(location, "main", true, xMember, Algorithm(location, { assignment0, assignment1 })));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 2> x = { 0, 0 };
	ArrayDescriptor<int, 1> xPtr(x.data(), { 2 });

	if (failed(runner.run("main", Runner::result(xPtr))))
		FAIL();

	EXPECT_EQ(xPtr[0], 23);
	EXPECT_EQ(xPtr[1], 57);
}

TEST(Output, integerArrayWithSizeDependingOnInputValue)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y[x + x];
	 *
	 *   algorithm
	 *     for i in 1 : (x + x)
	 *       y[i] := i;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Expression xReference = Expression::reference(location, makeType<int>(), "x");
	Expression ySize = Expression::operation(location, makeType<int>(), OperationKind::add, xReference, xReference);
	Member yMember(location, "y", makeType<int>(ySize), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement forAssignment = AssignmentStatement(
			location,
			Expression::operation(location, makeType<int>(), OperationKind::subscription,
														Expression::reference(location, makeType<int>(), "y"),
														Expression::reference(location, makeType<int>(), "i")),
			Expression::reference(location, makeType<int>(), "i"));

	Statement forStatement = ForStatement(
			location,
			Induction(
					"i",
					Expression::constant(location, makeType<int>(), 1),
					ySize),
			forAssignment);

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, forStatement)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	module.dump();

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	module.dump();

	Runner runner(module);

	int x = 2;
	ArrayDescriptor<float, 1> yPtr(nullptr, { 2 });

	if (failed(runner.run("main", x, Runner::result(yPtr))))
		FAIL();

	for (int i = 0; i < yPtr.getSize(0); i++)
		EXPECT_EQ(yPtr[i], i);
}

TEST(Output, floatArray)	 // NOLINT
{
	/**
	 * function main
	 *   output Real x[2];
	 *
	 *   algorithm
	 *     x[1] := 23;
	 *     x[2] := 57;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment0 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<float>(), OperationKind::subscription,
														Expression::reference(location, makeType<float>(), "x"),
														Expression::constant(location, makeType<int>(), 0)),
			Expression::constant(location, makeType<float>(), 23.0));

	Statement assignment1 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<float>(), OperationKind::subscription,
														Expression::reference(location, makeType<float>(), "x"),
														Expression::constant(location, makeType<int>(), 1)),
			Expression::constant(location, makeType<float>(), 57.0));

	ClassContainer cls(Function(location, "main", true, xMember, Algorithm(location, { assignment0, assignment1 })));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<float, 2> x = { 0, 0 };
	ArrayDescriptor<float, 1> xPtr(x.data(), { 2 });

	if (failed(runner.run("main", Runner::result(xPtr))))
		FAIL();

	EXPECT_FLOAT_EQ(xPtr[0], 23.0);
	EXPECT_FLOAT_EQ(xPtr[1], 57.0);
}

TEST(Output, integerMatrix)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer x[2,3];
	 *
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

	Member xMember(location, "x", makeType<int>(2, 3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	llvm::SmallVector<Statement, 3> assignments;

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 3; j++)
		{
			assignments.push_back(AssignmentStatement(
					location,
					Expression::operation(location, makeType<int>(), OperationKind::subscription,
																Expression::reference(location, makeType<int>(), "x"),
																Expression::constant(location, makeType<int>(), i),
																Expression::constant(location, makeType<int>(), j)),
					Expression::constant(location, makeType<int>(), i * 3 + j + 1)));
		}

	ClassContainer cls(Function(location, "main", true, xMember, Algorithm(location, assignments)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 6> x = { 0, 0, 0, 0, 0, 0 };
	ArrayDescriptor<int, 2> xPtr(x.data(), { 2, 3 });

	if (failed(runner.run("main", Runner::result(xPtr))))
		FAIL();

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 3; j++)
			EXPECT_EQ(xPtr.get(i, j), i * 3 + j + 1);
}
