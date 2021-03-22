#include <gtest/gtest.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/Dialect.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/Passes.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/utils/CRunnerUtils.h>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
using namespace std;

TEST(Function, callNoArguments)	 // NOLINT
{
	/**
	 * function foo
	 *   output Integer x;
	 *
	 *   algorithm
	 *   	 x := 1;
	 * end foo
	 *
	 * function main
	 *   output Integer x;
	 *
	 *   algorithm
	 *     x := foo();
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();
	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression xRef = Expression::reference(location, makeType<int>(), "x");

	Algorithm fooAlgorithm = Algorithm(location, {
			AssignmentStatement(location, xRef, Expression::constant(location, makeType<int>(), 1))
	});

	ClassContainer foo(Function(location, "foo", true, xMember, fooAlgorithm));

	Algorithm mainAlgorithm = Algorithm(location, {
			AssignmentStatement(location, xRef, Expression::call(location, makeType<int>(), Expression::reference(location, makeType<int>(), "foo")))
	});

	ClassContainer main(Function(location, "main", true, xMember, mainAlgorithm));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower({ foo, main });

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	int y = 0;

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", Runner::result(y))));
	EXPECT_EQ(y, 1);
}

TEST(Function, recursiveCall)	 // NOLINT
{
	/**
	 * function main
	 *   input Real[3] x;
	 *   input Integer i;
	 *   output Real y;
	 *
	 *   algorithm
	 *     y := 0
	 *     if i <= 3 then
	 *       y := x[i] + main(x, i + 1);
	 *     end if;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member iMember(location, "i", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Expression xRef = Expression::reference(location, makeType<float>(3), "x");
	Expression iRef = Expression::reference(location, makeType<int>(), "i");
	Expression yRef = Expression::reference(location, makeType<float>(), "y");

	Expression condition = Expression::operation(location, makeType<bool>(), OperationKind::lessEqual, iRef, Expression::constant(location, makeType<int>(), 3));

	Expression sum = Expression::operation(
			location, makeType<float>(), OperationKind::add,
	    Expression::operation(location, makeType<float>(), OperationKind::subscription, xRef,
														Expression::operation(location, makeType<int>(), OperationKind::subtract, iRef, Expression::constant(location, makeType<int>(), 1))),
			Expression::call(location, makeType<float>(), Expression::reference(location, makeType<float>(), "main"),
											 xRef,
											 Expression::operation(location, makeType<int>(), OperationKind::add, iRef, Expression::constant(location, makeType<int>(), 1))));

	Statement ifStatement = IfStatement(location, IfStatement::Block(condition, { AssignmentStatement(location, yRef, sum) }));

	Algorithm algorithm = Algorithm(location, {
			AssignmentStatement(location, yRef, Expression::constant(location, makeType<int>(), 0)),
			ifStatement
	});

	ClassContainer cls(Function(
			location, "main", true,
			{ xMember, yMember, iMember },
			algorithm));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<float, 3> x = { 1, 2, 3 };
	int i = 1;
	float y = 0;

	ArrayDescriptor<float, 1> xPtr(x.data(), { x.size() });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, i, Runner::result(y))));

	float expected = 0;

	for (float value : x)
		expected += value;

	EXPECT_EQ(y, expected);
}

TEST(Function, callWithStaticArrayAsOutput)	 // NOLINT
{
	/**
	 * function foo
	 *   output Integer[3] x;
	 *
	 *   algorithm
	 *   	 x[1] := 1;
	 *   	 x[2] := 2;
	 *   	 x[3] := 3;
	 * end foo
	 *
	 * function main
	 *   output Integer[3] x;
	 *
	 *   algorithm
	 *     x := foo();
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();
	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Expression xRef = Expression::reference(location, makeType<int>(3), "x");

	Algorithm fooAlgorithm = Algorithm(
			location,
			{
					AssignmentStatement(location,
															Expression::operation(location, makeType<int>(), OperationKind::subscription,
																										xRef,
																										Expression::constant(location, makeType<int>(), 0)),
															Expression::constant(location, makeType<int>(), 1)),
					AssignmentStatement(location,
															Expression::operation(location, makeType<int>(), OperationKind::subscription,
																										xRef,
																										Expression::constant(location, makeType<int>(), 1)),
															Expression::constant(location, makeType<int>(), 2)),
					AssignmentStatement(location,
															Expression::operation(location, makeType<int>(), OperationKind::subscription,
																										xRef,
																										Expression::constant(location, makeType<int>(), 2)),
															Expression::constant(location, makeType<int>(), 3))
			});

	ClassContainer foo(Function(location, "foo", true, xMember, fooAlgorithm));

	Algorithm mainAlgorithm = Algorithm(location, {
			AssignmentStatement(location, xRef, Expression::call(location, makeType<int>(3), Expression::reference(location, makeType<int>(3), "foo")))
	});

	ClassContainer main(Function(location, "main", true, xMember, mainAlgorithm));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower({ main, foo });

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 0, 0, 0 };
	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr)));
	EXPECT_EQ(xPtr[0], 1);
	EXPECT_EQ(xPtr[1], 2);
	EXPECT_EQ(xPtr[2], 3);
}

TEST(Function, callWithDynamicArrayAsOutput)	 // NOLINT
{
	/**
	 * function foo
	 *   input Real x;
	 *   input Integer n;
	 *   output Real y[n];
	 *
	 *   algorithm
	 *     for i in 1:n loop
	 *   	 	 y[i] := x * i;
	 *   	 end for;
	 * end foo
	 *
	 * function main
	 *   input
	 *   output Real[3] x;
	 *
	 *   algorithm
	 *     x := foo(2, 3);
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	llvm::SmallVector<Member, 3> fooMembers;
	fooMembers.emplace_back(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	fooMembers.emplace_back(location, "n", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	fooMembers.emplace_back(location, "y", makeType<float>(Expression::reference(location, makeType<int>(), "n")), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Induction induction("i", Expression::constant(location, makeType<int>(), 1), Expression::reference(location, makeType<int>(), "n"));

	Algorithm fooAlgorithm = Algorithm(
			location,
			{
					ForStatement(location, induction,
											 {
													 AssignmentStatement(location,
																							 Expression::operation(location, makeType<float>(), OperationKind::subscription,
																																		 Expression::reference(location, makeType<float>(-1), "y"),
																																		 Expression::operation(location, makeType<int>(), OperationKind::subtract,
																																													 Expression::reference(location, makeType<int>(), "i"),
																																													 Expression::constant(location, makeType<int>(), 1))),
																							 Expression::operation(location, makeType<float>(), OperationKind::multiply,
																																		 Expression::reference(location, makeType<float>(), "x"),
																																		 Expression::reference(location, makeType<int>(), "i")))
											 })
			});

	ClassContainer foo(Function(location, "foo", true, fooMembers, fooAlgorithm));

	llvm::SmallVector<Member, 3> mainMembers;
	mainMembers.emplace_back(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Algorithm mainAlgorithm = Algorithm(
			location,
			{
					AssignmentStatement(location,
															Expression::reference(location, makeType<float>(-1), "x"),
															Expression::call(location, makeType<float>(-1),
															    						 Expression::reference(location, makeType<float>(-1), "foo"),
																							 Expression::constant(location, makeType<float>(), 2),
																							 Expression::constant(location, makeType<int>(), 3)))
			});

	ClassContainer main(Function(location, "main", true, mainMembers, mainAlgorithm));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower({ foo, main });

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<float, 3> x = { 0, 0, 0 };
	ArrayDescriptor<float, 1> xPtr(x.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", Runner::result(xPtr))));
	EXPECT_FLOAT_EQ(xPtr[0], 2);
	EXPECT_FLOAT_EQ(xPtr[1], 4);
	EXPECT_FLOAT_EQ(xPtr[2], 6);
}

TEST(Function, callElementWise)	 // NOLINT
{
	/**
	 * function foo
	 *   input Integer x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *   	 y := -x;
	 * end foo
	 *
	 * function main
	 *   input Integer[3] x;
	 *   output Integer[3] y;
	 *
	 *   algorithm
	 *     y := foo(x);
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Algorithm fooAlgorithm = Algorithm(
			location,
			{
					AssignmentStatement(location,
															Expression::reference(location, makeType<int>(), "y"),
															Expression::operation(location, makeType<int>(), OperationKind::subtract,
																										Expression::reference(location, makeType<int>(), "x")))
			});

	ClassContainer foo(Function(location, "foo", true,
															{
																	Member(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input)),
																	Member(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output))
															},
															fooAlgorithm));

	Algorithm mainAlgorithm = Algorithm(
			location,
			{
					AssignmentStatement(location,
															Expression::reference(location, makeType<int>(3), "y"),
															Expression::callElementWise(location, makeType<int>(3), 1,
															    												Expression::reference(location, makeType<int>(), "foo"),
																													Expression::reference(location, makeType<int>(3), "x")))
	});

	ClassContainer main(Function(location, "main", true,
															 {
																	 Member(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input)),
																	 Member(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output))
															 },
															 mainAlgorithm));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower({ main, foo });

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 1, 0, 0 };
	array<int, 3> y = { 0, 0, 0 };
	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, Runner::result(yPtr))));
	EXPECT_EQ(yPtr[0], -1 * xPtr[0]);
	EXPECT_EQ(yPtr[1], -1 * xPtr[1]);
	EXPECT_EQ(yPtr[2], -1 * xPtr[2]);
}
