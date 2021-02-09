#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/mlirlowerer/MlirLowerer.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
using namespace std;

TEST(NegateOp, negateScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Boolean x;
	 *   output Boolean y;
	 *   algorithm
	 *     y := not x;
	 * end main
	 */

	/*
	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "y"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::negate,
														Expression::reference(location, makeType<BuiltInType::Boolean>(), "x")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	module.dump();

	if (failed(convertToLLVMDialect(&context, module)))
			FAIL();

	module.dump();

	Runner runner(&context, module);

	for (bool x : { true, false })
	{
		bool y = x;
		runner.run("main", x, y);
		EXPECT_EQ(y, !x);
	}
	 */
}

TEST(NegateOp, negateVector)	 // NOLINT
{
	/**
	 * function main
	 *   input Boolean[2] x;
	 *   output Boolean[2] y;
	 *   algorithm
	 *     y := not x;
	 * end main
	 */

	/*
	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Boolean>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Boolean>(2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(2), "y"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(2), OperationKind::negate,
														Expression::reference(location, makeType<BuiltInType::Boolean>(2), "x")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	module.dump();

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	module.dump();

	Runner runner(&context, module);

	/*
	array<bool, 2> x = { false, true };
	array<bool, 2> y = { true, false };

	bool* xPtr = x.data();
	bool* yPtr = y.data();
	 */

	/*
	bool x[2] = { true, false };
	bool y[2] = { true, false };

	bool *xPtr = &x[0];
	bool *yPtr = &y[0];

	runner.run("main", xPtr, yPtr);

	EXPECT_EQ(y[0], false);
	EXPECT_EQ(y[1], true);
	 */

	//for (const auto& pair : llvm::zip(x, y))
	//	EXPECT_EQ(get<1>(pair), !get<0>(pair));
}

TEST(AndOp, scalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Boolean x;
	 *   input Boolean y;
	 *   output Boolean z;
	 *   algorithm
	 *     z := x and y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::land,
														Expression::reference(location, makeType<BuiltInType::Boolean>(), "x"),
														Expression::reference(location, makeType<BuiltInType::Boolean>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<bool, 4> xData = { false, false, true, true };
	array<bool, 4> yData = { false, true, false, true };
	array<bool, 4> zData = { true, true, true, false };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		bool x = get<0>(tuple);
		bool y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x && y);
	}
}

TEST(OrOp, scalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Boolean x;
	 *   input Boolean y;
	 *   output Boolean z;
	 *   algorithm
	 *     z := x or y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Boolean>(), "z"),
			Expression::operation(location, makeType<BuiltInType::Boolean>(), OperationKind::lor,
														Expression::reference(location, makeType<BuiltInType::Boolean>(), "x"),
														Expression::reference(location, makeType<BuiltInType::Boolean>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	Runner runner(&context, module);

	array<bool, 4> xData = { false, false, true, true };
	array<bool, 4> yData = { false, true, false, true };
	array<bool, 4> zData = { true, false, false, false };

	for (const auto& tuple : llvm::zip(xData, yData, zData))
	{
		bool x = get<0>(tuple);
		bool y = get<1>(tuple);
		bool z = get<2>(tuple);

		runner.run("main", x, y, z);

		EXPECT_EQ(z, x || y);
	}
}
