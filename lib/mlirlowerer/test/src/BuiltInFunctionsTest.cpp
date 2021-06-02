#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <modelica/frontend/AST.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/runtime/ArrayDescriptor.h>
#include <numeric>

using namespace modelica;
using namespace frontend;
using namespace codegen;
using namespace std;

TEST(BuiltInOps, sizeSpecificArrayDimension)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[3, 2] x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := size(x, 2);
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(3, 2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::call(location, makeType<int>(),
											 Expression::reference(location, makeType<int>(), "size"),
											 llvm::ArrayRef({
													 Expression::reference(location, makeType<int>(3, 2), "x"),
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

	array<int, 6> x = { 1, 2, 3, 4, 5, 6 };
	ArrayDescriptor<int, 2> xDesc(x.data(), { 3, 2 });

	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, jit::Runner::result(y))));
	EXPECT_EQ(y, xDesc.getDimensionSize(1));
}

TEST(BuiltInOps, sizeAllArrayDimensions)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[4, 3] x;
	 *   output Integer[2] y;
	 *
	 *   algorithm
	 *     y := size(x);
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(4, 3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(2), "y"),
			Expression::call(location, makeType<int>(2),
											 Expression::reference(location, makeType<int>(2), "size"),
											 Expression::reference(location, makeType<int>(4, 3), "x")));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	array<int, 12> x = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	ArrayDescriptor<int, 2> xDesc(x.data(), { 4, 3 });

	array<int, 2> y = { 0, 0 };
	ArrayDescriptor<int, 1> yDesc(y);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, yDesc)));

	EXPECT_EQ(yDesc[0], xDesc.getDimensionSize(0));
	EXPECT_EQ(yDesc[1], xDesc.getDimensionSize(1));
}

TEST(BuiltInOps, sumOfIntegerArrayValues)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[3] x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := sum(x);
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::call(location, makeType<int>(),
			    Expression::reference(location, makeType<int>(), "sum"),
											 Expression::reference(location, makeType<int>(3), "x")));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);
	
	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	jit::Runner runner(*module);

	array<int, 3> x = { 1, 2, 3 };
	ArrayDescriptor<int, 1> xDesc(x);

	int y = 0;

	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, jit::Runner::result(y))));
	EXPECT_EQ(y, std::accumulate(x.begin(), x.end(), 0, std::plus<>()));
}

TEST(BuiltInOps, sumOfFloatArrayValues)	 // NOLINT
{
	/**
	 * function main
	 *   input Real[3] x;
	 *   output Real y;
	 *
	 *   algorithm
	 *     y := sum(x);
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "y"),
			Expression::call(location, makeType<float>(),
											 Expression::reference(location, makeType<float>(), "sum"),
											 Expression::reference(location, makeType<float>(3), "x")));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	jit::Runner runner(*module);

	array<float, 3> x = { 1, 2, 3 };
	ArrayDescriptor<float, 1> xDesc(x);

	float y = 0;

	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, jit::Runner::result(y))));
	EXPECT_EQ(y, std::accumulate(x.begin(), x.end(), 0, std::plus<>()));
}

TEST(BuiltInOps, productOfIntegerArrayValues)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[3] x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := product(x);
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::call(location, makeType<int>(),
											 Expression::reference(location, makeType<int>(), "product"),
											 Expression::reference(location, makeType<int>(3), "x")));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	jit::Runner runner(*module);

	array<int, 3> x = { 1, 2, 3 };
	ArrayDescriptor<int, 1> xDesc(x);

	int y = 0;

	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, jit::Runner::result(y))));
	EXPECT_EQ(y, std::accumulate(x.begin(), x.end(), 1, std::multiplies<>()));
}

TEST(BuiltInOps, productOfFloatArrayValues)	 // NOLINT
{
	/**
	 * function main
	 *   input Real[3] x;
	 *   output Real y;
	 *
	 *   algorithm
	 *     y := product(x);
	 * end main
	 */

	SourceRange location = SourceRange::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "y"),
			Expression::call(location, makeType<float>(),
											 Expression::reference(location, makeType<float>(), "product"),
											 Expression::reference(location, makeType<float>(3), "x")));

	auto cls = Class::standardFunction(
			location, true, "main",
			llvm::ArrayRef({ std::move(xMember), std::move(yMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;
	MLIRLowerer lowerer(context);

	auto module = lowerer.run(cls);
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, ModelicaLoweringOptions::testsOptions())));

	jit::Runner runner(*module);

	array<float, 3> x = { 1, 2, 3 };
	ArrayDescriptor<float, 1> xDesc(x);

	float y = 0;

	ASSERT_TRUE(mlir::succeeded(runner.run("main", xDesc, jit::Runner::result(y))));
	EXPECT_EQ(y, std::accumulate(x.begin(), x.end(), 1, std::multiplies<>()));
}