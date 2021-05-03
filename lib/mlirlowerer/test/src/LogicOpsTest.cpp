#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <modelica/frontend/AST.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/runtime/ArrayDescriptor.h>
#include <modelica/utils/SourcePosition.h>

using namespace modelica;
using namespace frontend;
using namespace codegen;
using namespace std;

TEST(Logic, negateScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Boolean x;
	 *   output Boolean y;
	 *
	 *   algorithm
	 *     y := not x;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "y"),
			Expression::operation(location, makeType<bool>(), OperationKind::negate,
														Expression::reference(location, makeType<bool>(), "x")));

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

	array<bool, 2> x = { true, false };
	array<bool, 2> y = { true, false };

	jit::Runner runner(*module);

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(y))));
		EXPECT_EQ(y, !x);
	}
}

TEST(Logic, negateArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Boolean[2] x;
	 *   output Boolean[2] y;
	 *
	 *   algorithm
	 *     y := not x;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<bool>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<bool>(2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(2), "y"),
			Expression::operation(location, makeType<bool>(2), OperationKind::negate,
														Expression::reference(location, makeType<bool>(2), "x")));

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

	array<bool, 2> x = { true, false };
	ArrayDescriptor<bool, 1> xPtr(x);

	array<bool, 2> y = { true, false };
	ArrayDescriptor<bool, 1> yPtr(y);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr)));

	for (const auto& [x, y] : llvm::zip(xPtr, yPtr))
		EXPECT_EQ(y, !x);
}

TEST(Logic, andScalars)	 // NOLINT
{
	/**
	 * function main
	 *   input Boolean x;
	 *   input Boolean y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x and y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::land,
														Expression::reference(location, makeType<bool>(), "x"),
														Expression::reference(location, makeType<bool>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<bool, 4> x = { false, false, true, true };
	array<bool, 4> y = { false, true, false, true };
	array<bool, 4> z = { true, true, true, false };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x && y);
	}
}

TEST(Logic, andArrays)	 // NOLINT
{
	/**
	 * function main
	 *   input Boolean[4] x;
	 *   input Boolean[4] y;
	 *   output Boolean[4] z;
	 *
	 *   algorithm
	 *     z := x and y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<bool>(4), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<bool>(4), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(4), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(4), "z"),
			Expression::operation(location, makeType<bool>(4), OperationKind::land,
														Expression::reference(location, makeType<bool>(4), "x"),
														Expression::reference(location, makeType<bool>(4), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<bool, 4> x = { false, false, true, true };
	ArrayDescriptor<bool, 1> xPtr(x);

	array<bool, 4> y = { false, true, false, true };
	ArrayDescriptor<bool, 1> yPtr(y);

	array<bool, 4> z = { true, true, true, false };
	ArrayDescriptor<bool, 1> zPtr(z);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, zPtr)));

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
		EXPECT_EQ(z, x && y);
}

TEST(Logic, orScalars)	 // NOLINT
{
	/**
	 * function main
	 *   input Boolean x;
	 *   input Boolean y;
	 *   output Boolean z;
	 *
	 *   algorithm
	 *     z := x or y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(), "z"),
			Expression::operation(location, makeType<bool>(), OperationKind::lor,
														Expression::reference(location, makeType<bool>(), "x"),
														Expression::reference(location, makeType<bool>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<bool, 4> x = { false, false, true, true };
	array<bool, 4> y = { false, true, false, true };
	array<bool, 4> z = { true, false, false, false };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, x || y);
	}
}

TEST(Logic, orArrays)	 // NOLINT
{
	/**
	 * function main
	 *   input Boolean[4] x;
	 *   input Boolean[4] y;
	 *   output Boolean[4] z;
	 *
	 *   algorithm
	 *     z := x or y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<bool>(4), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<bool>(4), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<bool>(4), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<bool>(4), "z"),
			Expression::operation(location, makeType<bool>(4), OperationKind::lor,
														Expression::reference(location, makeType<bool>(4), "x"),
														Expression::reference(location, makeType<bool>(4), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<bool, 4> x = { false, false, true, true };
	ArrayDescriptor<bool, 1> xPtr(x);

	array<bool, 4> y = { false, true, false, true };
	ArrayDescriptor<bool, 1> yPtr(y);

	array<bool, 4> z = { true, false, false, false };
	ArrayDescriptor<bool, 1> zPtr(z);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, zPtr)));

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
		EXPECT_EQ(z, x || y);
}
