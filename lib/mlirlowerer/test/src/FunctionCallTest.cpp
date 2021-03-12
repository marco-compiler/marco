#include <gtest/gtest.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/Dialect.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/frontend/Parser.hpp>
#include <modelica/mlirlowerer/MlirLowerer.h>
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
	 *   output x;
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
	MlirLowerer lowerer(context);
	auto module = lowerer.lower({ foo, main });

	if (!module || failed(convertToLLVMDialect(&context, *module)))
		FAIL();

	Runner runner(*module);

	int y = 0;

	if (failed(runner.run("main", Runner::result(y))))
		FAIL();

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
	MlirLowerer lowerer(context);
	auto module = lowerer.lower(cls);

	if (!module || failed(convertToLLVMDialect(&context, *module)))
		FAIL();

	Runner runner(*module);

	array<float, 3> x = { 1, 2, 3 };
	int i = 1;
	float y = 0;

	ArrayDescriptor<float, 1> xPtr(x.data(), { x.size() });

	if (failed(runner.run("main", xPtr, i, Runner::result(y))))
		FAIL();

	float expected = 0;

	for (float value : x)
		expected += value;

	EXPECT_EQ(y, expected);
}
