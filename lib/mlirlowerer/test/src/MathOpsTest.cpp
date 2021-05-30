#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <modelica/frontend/AST.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/runtime/ArrayDescriptor.h>

using namespace modelica;
using namespace frontend;
using namespace codegen;
using namespace std;

TEST(MathOps, negateIntegerScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := -x;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::operation(location, makeType<int>(), OperationKind::subtract,
														Expression::reference(location, makeType<int>(), "x")));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 2> x = { 23, 57 };
	array<int, 2> y = { 23, 57 };

	jit::Runner runner(*module);

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
		EXPECT_EQ(y, -1 * x);
	}
}

TEST(MathOps, negateFloatScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   output Real y;
	 *
	 *   algorithm
	 *     y := -x;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "y"),
			Expression::operation(location, makeType<float>(), OperationKind::subtract,
														Expression::reference(location, makeType<float>(), "x")));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<float, 2> x = { 23, 57 };
	array<float, 2> y = { 23, 57 };

	jit::Runner runner(*module);

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
		EXPECT_EQ(y, -1 * x);
	}
}

TEST(MathOps, negateIntegerStaticArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[3] x;
	 *   output Integer[3] y;
	 *
	 *   algorithm
	 *     y := -x;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "y"),
			Expression::operation(location, makeType<int>(3), OperationKind::subtract,
														Expression::reference(location, makeType<int>(3), "x")));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 10, 23, 57 };
	ArrayDescriptor<int, 1> xDesc(x);

	array<int, 3> y = { 10, 23, 57 };
	ArrayDescriptor<int, 1> yDesc(y);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc)));

	for (const auto& [x, y] : llvm::zip(xDesc, yDesc))
		EXPECT_EQ(y, -1 * x);
}

TEST(MathOps, negateIntegerDynamicArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[:] x;
	 *   output Integer[:] y;
	 *
	 *   algorithm
	 *     y := -x;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "y"),
			Expression::operation(location, makeType<int>(-1), OperationKind::subtract,
														Expression::reference(location, makeType<int>(-1), "x")));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 10, 23, 57 };
	ArrayDescriptor<int, 1> xDesc(x);

	array<int, 3> y = { 10, 23, 57 };
	ArrayDescriptor<int, 1> yDesc(y);
	auto* yPtr = &yDesc;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(yPtr), xDesc)));

	for (const auto& [x, y] : llvm::zip(xDesc, yDesc))
		EXPECT_EQ(y, -1 * x);
}

TEST(MathOps, negateFloatStaticArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Real[3] x;
	 *   output Real[3] y;
	 *
	 *   algorithm
	 *     y := -x;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(3), "y"),
			Expression::operation(location, makeType<float>(3), OperationKind::subtract,
														Expression::reference(location, makeType<float>(3), "x")));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<float, 3> x = { 10, 23, 57 };
	ArrayDescriptor<float, 1> xDesc(x);

	array<float, 3> y = { 10, 23, 57 };
	ArrayDescriptor<float, 1> yDesc(y);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc)));

	for (const auto& [x, y] : llvm::zip(xDesc, yDesc))
		EXPECT_EQ(y, -1 * x);
}

TEST(MathOps, negateFloatDynamicArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Real[:] x;
	 *   output Real[:] y;
	 *
	 *   algorithm
	 *     y := -x;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(-1), "y"),
			Expression::operation(location, makeType<float>(-1), OperationKind::subtract,
														Expression::reference(location, makeType<float>(-1), "x")));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<float, 3> x = { 10, 23, 57 };
	ArrayDescriptor<float, 1> xDesc(x);

	array<float, 3> y = { 10, 23, 57 };
	ArrayDescriptor<float, 1> yDesc(y);
	auto* yPtr = &yDesc;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(yPtr), xDesc)));

	for (const auto& [x, y] : llvm::zip(xDesc, yDesc))
		EXPECT_FLOAT_EQ(y, -1 * x);
}

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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 2> x = { 23, 57 };
	array<int, 2> y = { 57, -23 };
	array<int, 2> z = { 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::add,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(3), "x"),
																Expression::reference(location, makeType<int>(3), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 10, 23, 57 };
	ArrayDescriptor<int, 1> xDesc(x);

	array<int, 3> y = { 10, 57, -23 };
	ArrayDescriptor<int, 1> yDesc(y);

	array<int, 3> z = { 0, 0, 0 };
	ArrayDescriptor<int, 1> zDesc(z);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc, zDesc)));

	for (const auto& [x, y, z] : llvm::zip(xDesc, yDesc, zDesc))
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "z"),
			Expression::operation(location, makeType<int>(-1), OperationKind::add,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(-1), "x"),
																Expression::reference(location, makeType<int>(-1), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 10, 23, 57 };
	ArrayDescriptor<int, 1> xDesc(x);

	array<int, 3> y = { 10, 57, -23 };
	ArrayDescriptor<int, 1> yDesc(y);

	array<int, 3> z = { 0, 0, 0 };
	ArrayDescriptor<int, 1> zDesc(z);
	auto* zPtr = &zDesc;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(zPtr), xDesc, yDesc)));

	for (const auto& [x, y, z] : llvm::zip(xDesc, yDesc, zDesc))
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::add,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<float, 2> x = { 23.2, 57.5 };
	array<float, 2> y = { 57.3, -23.7 };
	array<float, 2> z = { 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(3), "z"),
			Expression::operation(location, makeType<float>(3), OperationKind::add,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(3), "x"),
																Expression::reference(location, makeType<float>(3), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<float, 3> x = { 10.1, 23.3, 57.8 };
	ArrayDescriptor<float, 1> xDesc(x);

	array<float, 3> y = { 10.2, 57.3, -23.5 };
	ArrayDescriptor<float, 1> yDesc(y);

	array<float, 3> z = { 0, 0, 0 };
	ArrayDescriptor<float, 1> zDesc(z);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc, zDesc)));

	for (const auto& [x, y, z] : llvm::zip(xDesc, yDesc, zDesc))
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(-1), "z"),
			Expression::operation(location, makeType<float>(-1), OperationKind::add,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(-1), "x"),
																Expression::reference(location, makeType<float>(-1), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<float, 3> x = { 10.1, 23.3, 57.8 };
	ArrayDescriptor<float, 1> xDesc(x);

	array<float, 3> y = { 10.2, 57.3, -23.5 };
	ArrayDescriptor<float, 1> yDesc(y);

	array<float, 3> z = { 0, 0, 0 };
	ArrayDescriptor<float, 1> zDesc(z);
	auto* zPtr = &zDesc;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(zPtr), xDesc, yDesc)));

	for (const auto& [x, y, z] : llvm::zip(xDesc, yDesc, zDesc))
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::add,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 2, -3, -3 };
	array<float, 3> y = { -3.5, 5.2, -2 };
	array<float, 3> z = { 0, 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(3), "z"),
			Expression::operation(location, makeType<float>(3), OperationKind::add,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(3), "x"),
																Expression::reference(location, makeType<float>(3), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 2, -3, -3 };
	ArrayDescriptor<int, 1> xDesc(x);

	array<float, 3> y = { -3.5, 5.2, -2 };
	ArrayDescriptor<float, 1> yDesc(y);

	array<float, 3> z = { 0, 0, 0 };
	ArrayDescriptor<float, 1> zDesc(z);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc, zDesc)));

	for (const auto& [x, y, z] : llvm::zip(xDesc, yDesc, zDesc))
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto tMember = Member::build(location, "t", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "t"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(), "y"),
																Expression::reference(location, makeType<int>(), "z")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember), std::move(tMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 4, -7, -15 };
	array<int, 3> t = { 0, 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z, t] : llvm::zip(x, y, z, t))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, z, jit::Runner::result(t))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::subtract,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 2> x = { 23, 57 };
	array<int, 2> y = { 57, -23 };
	array<int, 2> z = { 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::subtract,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(3), "x"),
																Expression::reference(location, makeType<int>(3), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 10, 23, 57 };
	ArrayDescriptor<int, 1> xDesc(x);

	array<int, 3> y = { 10, 57, -23 };
	ArrayDescriptor<int, 1> yDesc(y);

	array<int, 3> z = { 0, 0, 0 };
	ArrayDescriptor<int, 1> zDesc(z);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc, zDesc)));

	for (const auto& [x, y, z] : llvm::zip(xDesc, yDesc, zDesc))
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "z"),
			Expression::operation(location, makeType<int>(-1), OperationKind::subtract,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(-1), "x"),
																Expression::reference(location, makeType<int>(-1), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 10, 23, 57 };
	ArrayDescriptor<int, 1> xDesc(x);

	array<int, 3> y = { 10, 57, -23 };
	ArrayDescriptor<int, 1> yDesc(y);

	array<int, 3> z = { 0, 0, 0 };
	ArrayDescriptor<int, 1> zDesc(z);
	auto* zPtr = &zDesc;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(zPtr), xDesc, yDesc)));

	for (const auto& [x, y, z] : llvm::zip(xDesc, yDesc, zDesc))
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::subtract,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<float, 2> x = { 23.2, 57.5 };
	array<float, 2> y = { 57.3, -23.7 };
	array<float, 2> z = { 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(3), "z"),
			Expression::operation(location, makeType<float>(3), OperationKind::subtract,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(3), "x"),
																Expression::reference(location, makeType<float>(3), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<float, 3> x = { 10.1, 23.3, 57.8 };
	ArrayDescriptor<float, 1> xDesc(x);

	array<float, 3> y = { 10.2, 57.3, -23.5 };
	ArrayDescriptor<float, 1> yDesc(y);

	array<float, 3> z = { 0, 0, 0 };
	ArrayDescriptor<float, 1> zDesc(z);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc, zDesc)));

	for (const auto& [x, y, z] : llvm::zip(xDesc, yDesc, zDesc))
		EXPECT_FLOAT_EQ(z, x - y);
}

TEST(MathOps, subOfFloatDynamicArrays)	 // NOLINT
{
	/**
	 * function main
	 *   input Real[:] x;
	 *   input Real[:] y;
	 *   output Real[:] z;
	 *
	 *   algorithm
	 *     z := x - y;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(-1), "z"),
			Expression::operation(location, makeType<float>(-1), OperationKind::subtract,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(-1), "x"),
																Expression::reference(location, makeType<float>(-1), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<float, 3> x = { 10.1, 23.3, 57.8 };
	ArrayDescriptor<float, 1> xDesc(x);

	array<float, 3> y = { 10.2, 57.3, -23.5 };
	ArrayDescriptor<float, 1> yDesc(y);

	array<float, 3> z = { 0, 0, 0 };
	ArrayDescriptor<float, 1> zDesc(z);
	auto* zPtr = &zDesc;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(zPtr), xDesc, yDesc)));

	for (const auto& [x, y, z] : llvm::zip(xDesc, yDesc, zDesc))
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::subtract,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 2, -3, -3 };
	array<float, 3> y = { -3.5, 5.2, -2 };
	array<float, 3> z = { 0, 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(3), "z"),
			Expression::operation(location, makeType<float>(3), OperationKind::subtract,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(3), "x"),
																Expression::reference(location, makeType<float>(3), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 2, -3, -3 };
	ArrayDescriptor<int, 1> xDesc(x);

	array<float, 3> y = { -3.5, 5.2, -2 };
	ArrayDescriptor<float, 1> yDesc(y);

	array<float, 3> z = { 0, 0, 0 };
	ArrayDescriptor<float, 1> zDesc(z);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc, zDesc)));

	for (const auto& [x, y, z] : llvm::zip(xDesc, yDesc, zDesc))
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto tMember = Member::build(location, "t", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "t"),
			Expression::operation(location, makeType<int>(), OperationKind::subtract,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(), "y"),
																Expression::reference(location, makeType<int>(), "z")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember), std::move(tMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 4, -7, -15 };
	array<int, 3> t = { 0, 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z, t] : llvm::zip(x, y, z, t))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, z, jit::Runner::result(t))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::multiply,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 2> x = { 2, 5 };
	array<int, 2> y = { 3, -3 };
	array<int, 2> z = { 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::multiply,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<float, 2> x = { 2.3, 5.7 };
	array<float, 2> y = { 23.57, -23.57 };
	array<float, 2> z = { 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::multiply,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 2, -3, -3 };
	array<float, 3> y = { -3.5, 5.2, -2 };
	array<float, 3> z = { 0, 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto tMember = Member::build(location, "t", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "t"),
			Expression::operation(location, makeType<int>(), OperationKind::multiply,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(), "y"),
																Expression::reference(location, makeType<int>(), "z")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember), std::move(tMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 4, -7, -15 };
	array<int, 3> t = { 0, 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z, t] : llvm::zip(x, y, z, t))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, z, jit::Runner::result(t))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::multiply,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(3), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	int x = 2;

	array<int, 3> y = { 3, -5, 0 };
	ArrayDescriptor<int, 1> yDesc(y);

	array<int, 3> z = { 0, 0, 0 };
	ArrayDescriptor<int, 1> zDesc(z);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, yDesc, zDesc)));

	for (const auto& [y, z] : llvm::zip(yDesc, zDesc))
		EXPECT_EQ(z, x * y);
}

TEST(MathOps, mulIntegerScalarAndIntegerDynamicArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer[:] y;
	 *   output Integer[:] z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "z"),
			Expression::operation(location, makeType<int>(-1), OperationKind::multiply,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(-1), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	int x = 2;

	array<int, 3> y = { 3, -5, 0 };
	ArrayDescriptor<int, 1> yDesc(y);

	array<int, 3> z = { 0, 0, 0 };
	ArrayDescriptor<int, 1> zDesc(z);
	auto* zPtr = &zDesc;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(zPtr), x, yDesc)));

	for (const auto& [y, z] : llvm::zip(yDesc, zDesc))
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::multiply,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(3), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 3, -5, 0 };
	ArrayDescriptor<int, 1> xDesc(x);

	int y = 2;

	array<int, 3> z = { 0, 0, 0 };
	ArrayDescriptor<int, 1> zDesc(z);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, y, zDesc)));

	for (const auto& [x, z] : llvm::zip(xDesc, zDesc))
		EXPECT_EQ(z, x * y);
}

TEST(MathOps, mulIntegerDynamicArrayAndIntegerScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[:] x;
	 *   input Integer y;
	 *   output Integer[:] z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "z"),
			Expression::operation(location, makeType<int>(-1), OperationKind::multiply,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(-1), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 3, -5, 0 };
	ArrayDescriptor<int, 1> xDesc(x);

	int y = 2;

	array<int, 3> z = { 0, 0, 0 };
	ArrayDescriptor<int, 1> zDesc(z);
	auto* zPtr = &zDesc;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(zPtr), xDesc, y)));

	for (const auto& [x, z] : llvm::zip(xDesc, zDesc))
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::multiply,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(3), "x"),
																Expression::reference(location, makeType<int>(3), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 3, 5, 2 };
	ArrayDescriptor<int, 1> xDesc(x);

	array<int, 3> y = { 7, -2, 3 };
	ArrayDescriptor<int, 1> yDesc(y);

	int z = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc, jit::Runner::result(z))));

	EXPECT_EQ(z, 17);
}

TEST(MathOps, mulIntegerStaticVectorAndIntegerStaticMatrix)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[4] x;
	 *   input Integer[4,3] y;
	 *   output Integer[3] z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(4), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(4, 3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::multiply,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(4), "x"),
																Expression::reference(location, makeType<int>(4, 3), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 4> x = { 1, 2, 3, 4 };
	ArrayDescriptor<int, 1> xDesc(x);

	array<int, 12> y = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	ArrayDescriptor<int, 2> yDesc(y.data(), { 4, 3 });

	array<int, 3> z = { 0, 0, 0 };
	ArrayDescriptor<int, 1> zDesc(z);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc, zDesc)));

	EXPECT_EQ(zDesc.getDimensionSize(0), 3);

	EXPECT_EQ(zDesc[0], 70);
	EXPECT_EQ(zDesc[1], 80);
	EXPECT_EQ(zDesc[2], 90);
}

TEST(MathOps, mulIntegerStaticMatrixAndIntegerStaticVector)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[4,3] x;
	 *   input Integer[3] y;
	 *   output Integer[4] z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(4, 3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(4), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(4), "z"),
			Expression::operation(location, makeType<int>(4), OperationKind::multiply,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(4, 3), "x"),
																Expression::reference(location, makeType<int>(3), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 12> x = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	ArrayDescriptor<int, 2> xDesc(x.data(), { 4, 3 });

	array<int, 3> y = { 1, 2, 3 };
	ArrayDescriptor<int, 1> yDesc(y);

	array<int, 4> z = { 0, 0, 0, 0 };
	ArrayDescriptor<int, 1> zDesc(z);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc, zDesc)));

	EXPECT_EQ(zDesc.getDimensionSize(0), 4);

	EXPECT_EQ(zDesc[0], 14);
	EXPECT_EQ(zDesc[1], 32);
	EXPECT_EQ(zDesc[2], 50);
	EXPECT_EQ(zDesc[3], 68);
}

TEST(MathOps, mulIntegerStaticMatrixes)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[2,3] x;
	 *   input Integer[3,2] y;
	 *   output Integer[2,2] z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(2, 3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(3, 2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(2, 2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(2, 2), "z"),
			Expression::operation(location, makeType<int>(2, 2), OperationKind::multiply,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(2, 3), "x"),
																Expression::reference(location, makeType<int>(3, 2), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 6> x = { 1, 2, 3, 4, 5, 6 };
	ArrayDescriptor<int, 2> xDesc(x.data(), { 2, 3 });

	array<int, 6> y = { 1, 2, 3, 4, 5, 6 };
	ArrayDescriptor<int, 2> yDesc(y.data(), { 3, 2 });

	array<int, 4> z = { 0, 0, 0, 0 };
	ArrayDescriptor<int, 2> zDesc(z.data(), { 2, 2 });

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc, zDesc)));

	EXPECT_EQ(zDesc.getDimensionSize(0), 2);
	EXPECT_EQ(zDesc.getDimensionSize(1), 2);

	EXPECT_EQ(zDesc.get(0, 0), 22);
	EXPECT_EQ(zDesc.get(0, 1), 28);
	EXPECT_EQ(zDesc.get(1, 0), 49);
	EXPECT_EQ(zDesc.get(1, 1), 64);
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::divide,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 2> x = { 6, 10 };
	array<int, 2> y = { 3, -5 };
	array<int, 2> z = { 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::divide,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<float, 2> x = { 10.8, 10 };
	array<float, 2> y = { 3.6, -3.2 };
	array<float, 2> z = { 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto tMember = Member::build(location, "t", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "t"),
			Expression::operation(location, makeType<int>(), OperationKind::divide,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(), "y"),
																Expression::reference(location, makeType<int>(), "z")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember), std::move(tMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 120, 50, 0 };
	array<int, 3> y = { 2, 5, 5 };
	array<int, 3> z = { -3, 2, 2 };
	array<int, 3> t = { 0, 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z, t] : llvm::zip(x, y, z, t))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, z, jit::Runner::result(t))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::divide,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 23, 10, -3 };
	array<float, 3> y = { -3.5, 3.2, -2 };
	array<float, 3> z = { 0, 0, 0 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
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

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::divide,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(3), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 10, -5, 0 };
	ArrayDescriptor<int, 1> xDesc(x);

	int y = 2;

	array<int, 3> z = { 0, 0, 0 };
	ArrayDescriptor<int, 1> zDesc(z);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, y, zDesc)));

	for (const auto& [x, z] : llvm::zip(xDesc, zDesc))
		EXPECT_EQ(z, x / y);
}

TEST(MathOps, divIntegerDynamicArrayAndIntegerScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[:] x;
	 *   input Integer y;
	 *   output Integer[:] z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "z"),
			Expression::operation(location, makeType<int>(-1), OperationKind::divide,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(-1), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 3> x = { 3, -5, 0 };
	ArrayDescriptor<int, 1> xDesc(x);

	int y = 2;
	ArrayDescriptor<int, 1> zDesc(nullptr, { 3 });
	auto* zPtr = &zDesc;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", jit::Runner::result(zPtr), xDesc, y)));

	for (const auto& [x, z] : llvm::zip(xDesc, zDesc))
		EXPECT_EQ(z, x / y);
}

TEST(MathOps, powScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Integer z;
	 *
	 *   algorithm
	 *     z := x ^ y;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::powerOf,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 4> x = { 3, 2, 4, 0 };
	array<int, 4> y = { 4, 0, 2, 3 };
	array<int, 4> z = { 0, 0, 0, 1 };

	jit::Runner runner(*module);

	for (const auto& [ x, y, z ] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, pow(x, y));
	}
}

TEST(MathOps, powOneExponent)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := x ^ 1;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::operation(location, makeType<int>(), OperationKind::powerOf,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::constant(location, makeType<int>(), 1)
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 4> x = { 3, 2, 4, 0 };
	array<int, 4> y = { 0, 0, 0, 1 };

	jit::Runner runner(*module);

	for (const auto& [ x, y ] : llvm::zip(x, y))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
		EXPECT_EQ(y, pow(x, 1));
	}
}

TEST(MathOps, powSquare)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := x ^ 2;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::operation(location, makeType<int>(), OperationKind::powerOf,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::constant(location, makeType<int>(), 2)
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 4> x = { 3, 2, 4, 0 };
	array<int, 4> y = { 0, 0, 0, 1 };

	jit::Runner runner(*module);

	for (const auto& [ x, y ] : llvm::zip(x, y))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
		EXPECT_EQ(y, pow(x, 2));
	}
}

TEST(MathOps, powSquareMatrix)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[2,2] x;
	 *   input Integer y;
	 *   output Integer[2,2] z;
	 *
	 *   algorithm
	 *     z := x ^ y;
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(2, 2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<int>(2, 2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(2, 2), "z"),
			Expression::operation(location, makeType<int>(2, 2), OperationKind::powerOf,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(2, 2), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 4> x = { 1, 2, 3, 4 };
	ArrayDescriptor<int, 2> xDesc(x.data(), { 2, 2 });

	int y = 3;

	array<int, 4> z = { 0, 0, 0, 0 };
	ArrayDescriptor<int, 2> zDesc(z.data(), { 2, 2 });

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, y, zDesc)));

	EXPECT_EQ(zDesc.get(0, 0), 37);
	EXPECT_EQ(zDesc.get(0, 1), 54);
	EXPECT_EQ(zDesc.get(1, 0), 81);
	EXPECT_EQ(zDesc.get(1, 1), 118);
}
