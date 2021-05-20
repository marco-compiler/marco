#include <gtest/gtest.h>
#include <mlir/IR/Dialect.h>
#include <modelica/frontend/AST.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/runtime/ArrayDescriptor.h>

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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "x"),
			Expression::constant(location, makeType<int>(), 57));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None, std::move(xMember),
			Algorithm::build(location, assignment));

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

	SourceRange location = SourceRange::unknown();

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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "y"),
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::reference(location, makeType<float>(), "x"));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, std::move(assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto tMember = Member::build(location, "t", makeType<int>(3, 2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment1 = Statement::assignmentStatement(
			location,
			Expression::operation(location, makeType<int>(2), OperationKind::subscription,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(3, 2), "t"),
																Expression::constant(location, makeType<int>(), 0)
														})),
			Expression::reference(location, makeType<int>(2), "x"));

	auto assignment2 = Statement::assignmentStatement(
			location,
			Expression::operation(location, makeType<int>(2), OperationKind::subscription,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(3, 2), "t"),
																Expression::constant(location, makeType<int>(), 1)
														})),
			Expression::reference(location, makeType<int>(2), "y"));

	auto assignment3 = Statement::assignmentStatement(
			location,
			Expression::operation(location, makeType<int>(2), OperationKind::subscription,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(3, 2), "t"),
																Expression::constant(location, makeType<int>(), 2)
														})),
			Expression::reference(location, makeType<int>(2), "z"));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember), std::move(tMember) }),
			Algorithm::build(location, llvm::ArrayRef({ std::move(assignment1), std::move(assignment2), std::move(assignment3) })));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 2> x = { 0, 1 };
	ArrayDescriptor<int, 1> xDesc(x);

	array<int, 2> y = { 2, 3 };
	ArrayDescriptor<int, 1> yDesc(y);

	array<int, 2> z = { 4, 5 };
	ArrayDescriptor<int, 1> zDesc(z);

	array<int, 6> t = { 0, 0, 0, 0, 0, 0 };
	ArrayDescriptor<int, 2> tDesc(t.data(), { 3, 2 });

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc, zDesc, tDesc)));

	EXPECT_EQ(tDesc.get(0, 0), x[0]);
	EXPECT_EQ(tDesc.get(0, 1), x[1]);
	EXPECT_EQ(tDesc.get(1, 0), y[0]);
	EXPECT_EQ(tDesc.get(1, 1), y[1]);
	EXPECT_EQ(tDesc.get(2, 0), z[0]);
	EXPECT_EQ(tDesc.get(2, 1), z[1]);
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(2), "y"),
			Expression::reference(location, makeType<int>(2), "x"));

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

	array<int, 2> y = { 0, 0 };
	ArrayDescriptor<int, 1> yDesc(y);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc)));

	for (const auto& [x, y] : llvm::zip(xDesc, yDesc))
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	auto zMember = Member::build(location, "z", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::none));

	auto algorithm = Algorithm::build(
			location,
			llvm::ArrayRef({
					Statement::assignmentStatement(
							location,
							Expression::operation(location, makeType<int>(), OperationKind::subscription,
																		llvm::ArrayRef({
																				Expression::reference(location, makeType<int>(2), "z"),
																				Expression::constant(location, makeType<int>(), 0)
																		})),
							Expression::operation(location, makeType<int>(), OperationKind::multiply,
																		llvm::ArrayRef({
																				Expression::reference(location, makeType<int>(2), "x"),
																				Expression::constant(location, makeType<int>(), 2)
																		}))),
					Statement::assignmentStatement(
							location,
							Expression::operation(location, makeType<int>(), OperationKind::subscription,
																		llvm::ArrayRef({
																				Expression::reference(location, makeType<int>(2), "z"),
																				Expression::constant(location, makeType<int>(), 1)
																		})),
							Expression::operation(location, makeType<int>(), OperationKind::add,
																		llvm::ArrayRef({
																				Expression::operation(location, makeType<int>(), OperationKind::subscription,
																															llvm::ArrayRef({
																																	Expression::reference(location, makeType<int>(2), "z"),
																																	Expression::constant(location, makeType<int>(), 0)
																															})),
																				Expression::constant(location, makeType<int>(), 1)
																		}))),
					Statement::assignmentStatement(
							location,
							Expression::reference(location, makeType<int>(), "y"),
							Expression::operation(location, makeType<int>(), OperationKind::subscription,
																		llvm::ArrayRef({
																				Expression::reference(location, makeType<int>(2), "z"),
																				Expression::constant(location, makeType<int>(), 1)
																		})))
			}));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			std::move(algorithm));

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
	EXPECT_EQ(y, x * 2 + 1);
}
