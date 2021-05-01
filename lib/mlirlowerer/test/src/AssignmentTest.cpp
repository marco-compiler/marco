#include <gtest/gtest.h>
#include <mlir/IR/Dialect.h>
#include <modelica/frontend/AST.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/runtime/ArrayDescriptor.h>
#include <modelica/utils/SourcePosition.h>

using namespace modelica;
using namespace frontend;
using namespace codegen;
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "x"),
			Expression::constant(location, makeType<int>(), 57));

	ClassContainer cls(Function(location, "main", true, xMember, Algorithm(location, assignment)));
	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);
	
	auto module = lowerer.lower(cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	int x = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(x))));
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::reference(location, makeType<int>(), "x"));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	int x = 57;
	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
	EXPECT_EQ(y, x);
}

TEST(Assignment, implicitCastIntegerToFloat)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Real y;
	 *
	 *   algorithm
	 *     y := x;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "y"),
			Expression::reference(location, makeType<int>(), "x"));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	int x = 57;
	float y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
	EXPECT_FLOAT_EQ(y, x);
}

TEST(Assignment, implicitCastFloatToInteger)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := x;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::reference(location, makeType<float>(), "x"));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	float x = 1.8;
	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
	EXPECT_EQ(y, (int) x);
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

	Member xMember(location, "x", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member tMember(location, "t", makeType<int>(3, 2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment1 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<int>(2), OperationKind::subscription,
														Expression::reference(location, makeType<int>(3, 2), "t"),
														Expression::constant(location, makeType<int>(), 0)),
			Expression::reference(location, makeType<int>(2), "x"));

	Statement assignment2 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<int>(2), OperationKind::subscription,
														Expression::reference(location, makeType<int>(3, 2), "t"),
														Expression::constant(location, makeType<int>(), 1)),
			Expression::reference(location, makeType<int>(2), "y"));

	Statement assignment3 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<int>(2), OperationKind::subscription,
														Expression::reference(location, makeType<int>(3, 2), "t"),
														Expression::constant(location, makeType<int>(), 2)),
			Expression::reference(location, makeType<int>(2), "z"));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember, tMember },
															Algorithm(location, { assignment1, assignment2, assignment3 })));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 2> x = { 0, 1 };
	array<int, 2> y = { 2, 3 };
	array<int, 2> z = { 4, 5 };
	array<int, 6> t = { 0, 0, 0, 0, 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 2 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 2 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 2 });
	ArrayDescriptor<int, 2> tPtr(t.data(), { 3, 2 });

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, zPtr, jit::Runner::result(tPtr))));

	EXPECT_EQ(tPtr.get(0, 0), x[0]);
	EXPECT_EQ(tPtr.get(0, 1), x[1]);
	EXPECT_EQ(tPtr.get(1, 0), y[0]);
	EXPECT_EQ(tPtr.get(1, 1), y[1]);
	EXPECT_EQ(tPtr.get(2, 0), z[0]);
	EXPECT_EQ(tPtr.get(2, 1), z[1]);
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

	Member xMember(location, "x", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(2), "y"),
			Expression::reference(location, makeType<int>(2), "x"));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 2> x = { 23, 57 };
	array<int, 2> y = { 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 2 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 2 });

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, jit::Runner::result(yPtr))));

	for (const auto& [x, y] : llvm::zip(xPtr, yPtr))
		EXPECT_EQ(y, x);
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

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Member zMember(location, "z", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::none));

	Algorithm algorithm(
			location,
			{
					AssignmentStatement(
							location,
							Expression::operation(location, makeType<int>(), OperationKind::subscription,
																		Expression::reference(location, makeType<int>(2), "z"),
																		Expression::constant(location, makeType<int>(), 0)),
							Expression::operation(location, makeType<int>(), OperationKind::multiply,
																		Expression::reference(location, makeType<int>(2), "x"),
																		Expression::constant(location, makeType<int>(), 2))),
					AssignmentStatement(
							location,
							Expression::operation(location, makeType<int>(), OperationKind::subscription,
																		Expression::reference(location, makeType<int>(2), "z"),
																		Expression::constant(location, makeType<int>(), 1)),
							Expression::operation(location, makeType<int>(), OperationKind::add,
																		Expression::operation(location, makeType<int>(), OperationKind::subscription,
																													Expression::reference(location, makeType<int>(2), "z"),
																													Expression::constant(location, makeType<int>(), 0)),
																		Expression::constant(location, makeType<int>(), 1))),
					AssignmentStatement(
							location,
							Expression::reference(location, makeType<int>(), "y"),
							Expression::operation(location, makeType<int>(), OperationKind::subscription,
																		Expression::reference(location, makeType<int>(2), "z"),
																		Expression::constant(location, makeType<int>(), 1)))
			});

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															algorithm));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	int x = 57;
	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
	EXPECT_EQ(y, x * 2 + 1);
}
