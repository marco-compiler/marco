#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/mlirlowerer/CRunnerUtils.h>
#include <modelica/mlirlowerer/MlirLowerer.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
using namespace std;

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

	/*
	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::call(location, makeType<int>(),
			    Expression::reference(location, makeType<int>(), "sum"),
											 Expression::reference(location, makeType<int>(3), "x")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(module);

	array<int, 3> x = { 10, 23, -57 };
	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });

	int y = 0;

	if (failed(runner.run("main", xPtr, Runner::result(y))))
		FAIL();

	EXPECT_EQ(y, x[0] + x[1] + x[2]);
	 */
}