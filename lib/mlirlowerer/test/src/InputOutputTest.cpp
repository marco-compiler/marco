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

	auto xMember = Member::build(location, "x", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "y"),
			Expression::reference(location, makeType<bool>(), "x"));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	bool x = true;
	bool y = false;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
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

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::reference(location, makeType<int>(), "x"));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	int x = 57;
	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
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

	auto xMember = Member::build(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "y"),
			Expression::reference(location, makeType<float>(), "x"));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	float x = 57;
	float y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
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

	auto xMember = Member::build(location, "x", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														llvm::ArrayRef({
																Expression::operation(location, makeType<int>(), OperationKind::subscription,
																											llvm::ArrayRef({
																													Expression::reference(location, makeType<int>(), "x"),
																													Expression::constant(location, makeType<int>(), 0)
																											})),
																Expression::operation(location, makeType<int>(), OperationKind::subscription,
																											llvm::ArrayRef({
																													Expression::reference(location, makeType<int>(), "x"),
																													Expression::constant(location, makeType<int>(), 1)
																											}))
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));
	
	array<int, 2> x = { 23, 57 };
	ArrayDescriptor<int, 1> xDesc(x);

	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, jit::Runner::result(y))));
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

	auto xMember = Member::build(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														llvm::ArrayRef({
																Expression::operation(location, makeType<int>(), OperationKind::subscription,
																											llvm::ArrayRef({
																													Expression::reference(location, makeType<int>(), "x"),
																													Expression::constant(location, makeType<int>(), 0)
																											})),
																Expression::operation(location, makeType<int>(), OperationKind::subscription,
																											llvm::ArrayRef({
																													Expression::reference(location, makeType<int>(), "x"),
																													Expression::constant(location, makeType<int>(), 1)
																											}))
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 5> x = { 23, 57, 10, -23, -10 };
	ArrayDescriptor<int, 1> xDesc(x);

	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, jit::Runner::result(y))));
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

	auto xMember = Member::build(location, "x", makeType<float>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "y"),
			Expression::operation(location, makeType<float>(), OperationKind::add,
														llvm::ArrayRef({
																Expression::operation(
																		location, makeType<float>(), OperationKind::subscription,
																		llvm::ArrayRef({
																				Expression::reference(location, makeType<float>(), "x"),
																				Expression::constant(location, makeType<int>(), 0)
																		})),
																Expression::operation(
																		location, makeType<float>(), OperationKind::subscription,
																		llvm::ArrayRef({
																				Expression::reference(location, makeType<float>(), "x"),
																				Expression::constant(location, makeType<int>(), 1)
																		}))
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<float, 2> x = { 23.0, 57.0 };
	ArrayDescriptor<float, 1> xPtr(x);

	float y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, jit::Runner::result(y))));
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

	auto xMember = Member::build(location, "x", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "y"),
			Expression::operation(location, makeType<float>(), OperationKind::add,
														llvm::ArrayRef({
																Expression::operation(
																		location, makeType<float>(), OperationKind::subscription,
																		llvm::ArrayRef({
																				Expression::reference(location, makeType<float>(), "x"),
																				Expression::constant(location, makeType<int>(), 0)
																		})),
																Expression::operation(
																		location, makeType<float>(), OperationKind::subscription,
																		llvm::ArrayRef({
																				Expression::reference(location, makeType<float>(), "x"),
																				Expression::constant(location, makeType<int>(), 1)
																		}))
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<float, 2> x = { 23.0, 57.0 };
	ArrayDescriptor<float, 1> xDesc(x);

	float y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, jit::Runner::result(y))));
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

	auto xMember = Member::build(location, "x", makeType<int>(2, 3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	auto zMember = Member::build(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto yAssignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														llvm::ArrayRef({
																Expression::operation(location, makeType<int>(), OperationKind::subscription,
																											llvm::ArrayRef({
																													Expression::reference(location, makeType<int>(), "x"),
																													Expression::constant(location, makeType<int>(), 0),
																													Expression::constant(location, makeType<int>(), 0)
																											})),
																Expression::operation(location, makeType<int>(), OperationKind::subscription,
																											llvm::ArrayRef({
																													Expression::reference(location, makeType<int>(), "x"),
																													Expression::constant(location, makeType<int>(), 0),
																													Expression::constant(location, makeType<int>(), 1)
																											})),
																Expression::operation(location, makeType<int>(), OperationKind::subscription,
																											llvm::ArrayRef({
																													Expression::reference(location, makeType<int>(), "x"),
																													Expression::constant(location, makeType<int>(), 0),
																													Expression::constant(location, makeType<int>(), 2)
																											}))
														})));

	auto zAssignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														llvm::ArrayRef({
																Expression::operation(location, makeType<int>(), OperationKind::subscription,
																											llvm::ArrayRef({
																													Expression::reference(location, makeType<int>(), "x"),
																													Expression::constant(location, makeType<int>(), 1),
																													Expression::constant(location, makeType<int>(), 0)
																											})),
																Expression::operation(location, makeType<int>(), OperationKind::subscription,
																											llvm::ArrayRef({
																													Expression::reference(location, makeType<int>(), "x"),
																													Expression::constant(location, makeType<int>(), 1),
																													Expression::constant(location, makeType<int>(), 1)
																											})),
																Expression::operation(location, makeType<int>(), OperationKind::subscription,
																											llvm::ArrayRef({
																													Expression::reference(location, makeType<int>(), "x"),
																													Expression::constant(location, makeType<int>(), 1),
																													Expression::constant(location, makeType<int>(), 2)
																											}))
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location,
											 llvm::ArrayRef({
													 std::move(yAssignment), std::move(zAssignment)
											 })));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 6> x = { 1, 2, 3, 4, 5, 6 };
	ArrayDescriptor<int, 2> xDesc(x.data(), { 2, 3 });

	struct {
		int y = 0;
		int z = 0;
	} result;

	auto* resultPtr = &result;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(resultPtr), xDesc)));

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

	auto xMember = Member::build(location, "x", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment0 = Statement::assignmentStatement(
			location,
			Expression::operation(location, makeType<int>(), OperationKind::subscription,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::operation(location, makeType<int>(), OperationKind::add,
																											llvm::ArrayRef({
																													Expression::constant(location, makeType<int>(), 1),
																													Expression::constant(location, makeType<int>(), -1)
																											}))
														})),
			Expression::constant(location, makeType<int>(), 23));

	auto assignment1 = Statement::assignmentStatement(
			location,
			Expression::operation(location, makeType<int>(), OperationKind::subscription,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::operation(location, makeType<int>(), OperationKind::add,
																											llvm::ArrayRef({
																													Expression::constant(location, makeType<int>(), 2),
																													Expression::constant(location, makeType<int>(), -1)
																											}))
														})),
			Expression::constant(location, makeType<int>(), 57));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			std::move(xMember),
			Algorithm::build(location,
											 llvm::ArrayRef({
													 std::move(assignment0), std::move(assignment1)
											 })));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 2> x = { 0, 0 };
	ArrayDescriptor<int, 1> xDesc(x);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc)));

	EXPECT_EQ(xDesc[0], 23);
	EXPECT_EQ(xDesc[1], 57);
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

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto xReference = Expression::reference(location, makeType<int>(), "x");

	auto ySize = Expression::operation(
			location, makeType<int>(), OperationKind::add,
			llvm::ArrayRef({ xReference->clone(), xReference->clone() }));

	auto yMember = Member::build(location, "y", makeType<int>(ySize->clone()), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto forAssignment = Statement::assignmentStatement(
			location,
			Expression::operation(location, makeType<int>(), OperationKind::subscription,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "y"),
																Expression::operation(location, makeType<int>(), OperationKind::add,
																											llvm::ArrayRef({
																													Expression::reference(location, makeType<int>(), "i"),
																													Expression::constant(location, makeType<int>(), -1)
																											}))
														})),
			Expression::reference(location, makeType<int>(), "i"));

	auto forStatement = Statement::forStatement(
			location,
			Induction::build(
					location, "i",
					Expression::constant(location, makeType<int>(), 1),
					ySize->clone()),
			forAssignment);

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, forStatement));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	int x = 2;
	ArrayDescriptor<int, 1> yDesc(nullptr, { 2 });
	auto* yPtr = &yDesc;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(yPtr), x)));

	for (size_t i = 0; i < yDesc.getDimensionSize(0); i++)
		EXPECT_EQ(yDesc[i], i + 1);
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

	auto xMember = Member::build(location, "x", makeType<float>(2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment0 = Statement::assignmentStatement(
			location,
			Expression::operation(location, makeType<float>(), OperationKind::subscription,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(), "x"),
																Expression::constant(location, makeType<int>(), 0)
														})),
			Expression::constant(location, makeType<float>(), 23.0));

	auto assignment1 = Statement::assignmentStatement(
			location,
			Expression::operation(location, makeType<float>(), OperationKind::subscription,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(), "x"),
																Expression::constant(location, makeType<int>(), 1)
														})),
			Expression::constant(location, makeType<float>(), 57.0));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			std::move(xMember),
			Algorithm::build(location, llvm::ArrayRef({
																		 std::move(assignment0),
																		 std::move(assignment1)
																 })));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<float, 2> x = { 0, 0 };
	ArrayDescriptor<float, 1> xDesc(x);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc)));

	EXPECT_FLOAT_EQ(xDesc[0], 23.0);
	EXPECT_FLOAT_EQ(xDesc[1], 57.0);
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

	auto xMember = Member::build(location, "x", makeType<int>(2, 3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	llvm::SmallVector<std::unique_ptr<Statement>, 3> assignments;

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 3; j++)
		{
			assignments.push_back(Statement::assignmentStatement(
					location,
					Expression::operation(location, makeType<int>(), OperationKind::subscription,
																llvm::ArrayRef({
																		Expression::reference(location, makeType<int>(), "x"),
																		Expression::constant(location, makeType<int>(), i),
																		Expression::constant(location, makeType<int>(), j)
																})),
					Expression::constant(location, makeType<int>(), i * 3 + j + 1)));
		}

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			std::move(xMember),
			Algorithm::build(location, assignments));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 6> x = { 0, 0, 0, 0, 0, 0 };
	ArrayDescriptor<int, 2> xDesc(x.data(), { 2, 3 });

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc)));

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 3; j++)
			EXPECT_EQ(xDesc.get(i, j), i * 3 + j + 1);
}

TEST(Output, defaultScalarValue)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer x = 1;
	 *
	 *   algorithm
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output), Expression::constant(location, makeType<int>(), 1));
	auto cls = Class::standardFunction(location, true, "main", llvm::None, xMember, Algorithm::build(location, llvm::None));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	int x = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(x))));

	EXPECT_EQ(x, 1);
}

TEST(Output, defaultArrayValue)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer[3] x = { 1, 2, 3 };
	 *
	 *   algorithm
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(
			location, "x", makeType<int>(3),
			TypePrefix(ParameterQualifier::none, IOQualifier::output),
			Expression::array(
					location, makeType<int>(3),
			    llvm::ArrayRef({
							Expression::constant(location, makeType<int>(), 1),
							Expression::constant(location, makeType<int>(), 2),
							Expression::constant(location, makeType<int>(), 3) })
					));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None, xMember,
			Algorithm::build(location, llvm::None));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 3> x = { 0, 0, 0 };
	ArrayDescriptor<int, 1> xDesc(x);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc)));

	EXPECT_EQ(xDesc[0], 1);
	EXPECT_EQ(xDesc[1], 2);
	EXPECT_EQ(xDesc[2], 3);
}
