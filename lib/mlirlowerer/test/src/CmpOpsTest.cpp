#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <modelica/frontend/AST.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/utils/SourcePosition.h>

using namespace modelica;
using namespace frontend;
using namespace codegen;
using namespace std;

TEST(Comparison, eqIntegers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x == y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::equal,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 2> x = { 57, 57 };
	array<int, 2> y = { 57, 23 };
	array<bool, 2> z = { false, true };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x == y);
	}
}

TEST(Comparison, eqFloats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x == y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::equal,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(), "x"),
																Expression::reference(location, makeType<float>(), "y")	
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<float, 2> x = { 57, 57 };
	array<float, 2> y = { 57, 23 };
	array<bool, 2> z = { false, true };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x == y);
	}
}

TEST(Comparison, eqIntegerAndFloat)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x == y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::equal,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 2> x = { 57, 57 };
	array<float, 2> y = { 57, 23 };
	array<bool, 2> z = { false, true };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x == y);
	}
}

TEST(Comparison, notEqIntegers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x != y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::different,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 2> x = { 57, 57 };
	array<int, 2> y = { 57, 23 };
	array<bool, 2> z = { true, false };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x != y);
	}
}

TEST(Comparison, notEqFloats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x != y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::different,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<float, 2> x = { 57, 57 };
	array<float, 2> y = { 57, 23 };
	array<bool, 2> z = { true, false };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x != y);
	}
}

TEST(Comparison, notEqIntegerAndFloat)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x != y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::different,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 2> x = { 57, 57 };
	array<float, 2> y = { 57, 23 };
	array<bool, 2> z = { true, false };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x != y);
	}
}

TEST(Comparison, gtIntegers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x > y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::greater,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 3> x = { 23, 57, 57 };
	array<int, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { true, true, false };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x > y);
	}
}

TEST(Comparison, gtFloats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x > y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::greater,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<float, 3> x = { 23, 57, 57 };
	array<float, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { true, true, false };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x > y);
	}
}

TEST(Comparison, gtIntegerAndFloat)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x > y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::greater,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 3> x = { 23, 57, 57 };
	array<float, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { true, true, false };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x > y);
	}
}

TEST(Comparison, gteIntegers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x >= y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::greaterEqual,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 3> x = { 23, 57, 57 };
	array<int, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { true, false, false };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x >= y);
	}
}

TEST(Comparison, gteFloats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x >= y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::greaterEqual,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<float, 3> x = { 23, 57, 57 };
	array<float, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { true, false, false };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x >= y);
	}
}

TEST(Comparison, gteIntegerAndFloat)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x >= y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::greaterEqual,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 3> x = { 23, 57, 57 };
	array<float, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { true, false, false };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x >= y);
	}
}

TEST(Comparison, ltIntegers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x < y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::less,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 3> x = { 23, 57, 57 };
	array<int, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { false, true, true };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x < y);
	}
}

TEST(Comparison, ltFloats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x < y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::less,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<float, 3> x = { 23, 57, 57 };
	array<float, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { false, true, true };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x < y);
	}
}

TEST(Comparison, ltIntegerAndFloat)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x < y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::less,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 3> x = { 23, 57, 57 };
	array<float, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { false, true, true };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x < y);
	}
}

TEST(Comparison, lteIntegers)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x <= y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::lessEqual,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<int>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 3> x = { 23, 57, 57 };
	array<int, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { false, false, true };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_FALSE(failed(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x <= y);
	}
}

TEST(Comparison, lteFloats)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x <= y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::lessEqual,
														llvm::ArrayRef({
																Expression::reference(location, makeType<float>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<float, 3> x = { 23, 57, 57 };
	array<float, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { false, false, true };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x <= y);
	}
}

TEST(Comparison, lteIntegerAndFloat)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Real y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x <= y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	auto xMember = Member::build(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto yMember = Member::build(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	auto zMember = Member::build(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	auto assignment = Statement::assignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::lessEqual,
														llvm::ArrayRef({
																Expression::reference(location, makeType<int>(), "x"),
																Expression::reference(location, makeType<float>(), "y")
														})));

	auto cls = Class::standardFunction(
			location, true, "main", llvm::None,
			llvm::ArrayRef({ std::move(xMember), std::move(yMember), std::move(zMember) }),
			Algorithm::build(location, assignment));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.run(*cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 3> x = { 23, 57, 57 };
	array<float, 3> y = { 57, 57, 23 };
	array<bool, 3> z = { false, false, true };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x <= y);
	}
}
