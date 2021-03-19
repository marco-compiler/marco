#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <modelica/frontend/AST.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/utils/CRunnerUtils.h>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::operation(location, makeType<int>(), OperationKind::subtract,
														Expression::reference(location, makeType<int>(), "x")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 2> x = { 23, 57 };
	array<int, 2> y = { 23, 57 };

	Runner runner(*module);

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, Runner::result(y))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "y"),
			Expression::operation(location, makeType<float>(), OperationKind::subtract,
														Expression::reference(location, makeType<float>(), "x")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<float, 2> x = { 23, 57 };
	array<float, 2> y = { 23, 57 };

	Runner runner(*module);

	for (const auto& [x, y] : llvm::zip(x, y))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, Runner::result(y))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "y"),
			Expression::operation(location, makeType<int>(3), OperationKind::subtract,
														Expression::reference(location, makeType<int>(3), "x")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 23, 57 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, Runner::result(yPtr))));

	for (const auto& [x, y] : llvm::zip(xPtr, yPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "y"),
			Expression::operation(location, makeType<int>(-1), OperationKind::subtract,
														Expression::reference(location, makeType<int>(-1), "x")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);
	module->dump();

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 23, 57 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, Runner::result(yPtr))));

	for (const auto& [x, y] : llvm::zip(xPtr, yPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(3), "y"),
			Expression::operation(location, makeType<float>(3), OperationKind::subtract,
														Expression::reference(location, makeType<float>(3), "x")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<float, 3> x = { 10, 23, 57 };
	array<float, 3> y = { 10, 23, 57 };

	ArrayDescriptor<float, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<float, 1> yPtr(y.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, Runner::result(yPtr))));

	for (const auto& [x, y] : llvm::zip(xPtr, yPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(-1), "y"),
			Expression::operation(location, makeType<float>(-1), OperationKind::subtract,
														Expression::reference(location, makeType<float>(-1), "x")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<float, 3> x = { 10, 23, 57 };
	array<float, 3> y = { 10, 23, 57 };

	ArrayDescriptor<float, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<float, 1> yPtr(y.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, Runner::result(yPtr))));

	for (const auto& [x, y] : llvm::zip(xPtr, yPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 2> x = { 23, 57 };
	array<int, 2> y = { 57, -23 };
	array<int, 2> z = { 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, Runner::result(z))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::add,
														Expression::reference(location, makeType<int>(3), "x"),
														Expression::reference(location, makeType<int>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);
	module->dump();

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, Runner::result(zPtr))));

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "z"),
			Expression::operation(location, makeType<int>(-1), OperationKind::add,
														Expression::reference(location, makeType<int>(-1), "x"),
														Expression::reference(location, makeType<int>(-1), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, Runner::result(zPtr))));

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::add,
														Expression::reference(location, makeType<float>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<float, 2> x = { 23.2, 57.5 };
	array<float, 2> y = { 57.3, -23.7 };
	array<float, 2> z = { 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, Runner::result(z))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(3), "z"),
			Expression::operation(location, makeType<float>(3), OperationKind::add,
														Expression::reference(location, makeType<float>(3), "x"),
														Expression::reference(location, makeType<float>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<float, 3> x = { 10.1, 23.3, 57.8 };
	array<float, 3> y = { 10.2, 57.3, -23.5 };
	array<float, 3> z = { 0, 0, 0 };

	ArrayDescriptor<float, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<float, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<float, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, Runner::result(zPtr))));

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(-1), "z"),
			Expression::operation(location, makeType<float>(-1), OperationKind::add,
														Expression::reference(location, makeType<float>(-1), "x"),
														Expression::reference(location, makeType<float>(-1), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<float, 3> x = { 10.1, 23.3, 57.8 };
	array<float, 3> y = { 10.2, 57.3, -23.5 };
	array<float, 3> z = { 0, 0, 0 };

	ArrayDescriptor<float, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<float, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<float, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, Runner::result(zPtr))));

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::add,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 2, -3, -3 };
	array<float, 3> y = { -3.5, 5.2, -2 };
	array<float, 3> z = { 0, 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, Runner::result(z))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(3), "z"),
			Expression::operation(location, makeType<float>(3), OperationKind::add,
														Expression::reference(location, makeType<int>(3), "x"),
														Expression::reference(location, makeType<float>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 2, -3, -3 };
	array<float, 3> y = { -3.5, 5.2, -2 };
	array<float, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<float, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<float, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, Runner::result(zPtr))));

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member tMember(location, "t", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "t"),
			Expression::operation(location, makeType<int>(), OperationKind::add,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y"),
														Expression::reference(location, makeType<int>(), "z")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember, tMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 4, -7, -15 };
	array<int, 3> t = { 0, 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z, t] : llvm::zip(x, y, z, t))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, z, Runner::result(t))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::subtract,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 2> x = { 23, 57 };
	array<int, 2> y = { 57, -23 };
	array<int, 2> z = { 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, Runner::result(z))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::subtract,
														Expression::reference(location, makeType<int>(3), "x"),
														Expression::reference(location, makeType<int>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, Runner::result(zPtr))));

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "z"),
			Expression::operation(location, makeType<int>(-1), OperationKind::subtract,
														Expression::reference(location, makeType<int>(-1), "x"),
														Expression::reference(location, makeType<int>(-1), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, Runner::result(zPtr))));

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::subtract,
														Expression::reference(location, makeType<float>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<float, 2> x = { 23.2, 57.5 };
	array<float, 2> y = { 57.3, -23.7 };
	array<float, 2> z = { 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, Runner::result(z))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(3), "z"),
			Expression::operation(location, makeType<float>(3), OperationKind::subtract,
														Expression::reference(location, makeType<float>(3), "x"),
														Expression::reference(location, makeType<float>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<float, 3> x = { 10.1, 23.3, 57.8 };
	array<float, 3> y = { 10.2, 57.3, -23.5 };
	array<float, 3> z = { 0, 0, 0 };

	ArrayDescriptor<float, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<float, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<float, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, Runner::result(zPtr))));

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
		EXPECT_FLOAT_EQ(z, x - y);
}

TEST(MathOps, subOfFloatDynamicArrays)	 // NOLINT
{
	/**
	 * function main
	 *   input Real[-1] x;
	 *   input Real[-1] y;
	 *   output Real[-1] z;
	 *
	 *   algorithm
	 *     z := x - y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(-1), "z"),
			Expression::operation(location, makeType<float>(-1), OperationKind::subtract,
														Expression::reference(location, makeType<float>(-1), "x"),
														Expression::reference(location, makeType<float>(-1), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<float, 3> x = { 10.1, 23.3, 57.8 };
	array<float, 3> y = { 10.2, 57.3, -23.5 };
	array<float, 3> z = { 0, 0, 0 };

	ArrayDescriptor<float, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<float, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<float, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, Runner::result(zPtr))));

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::subtract,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 2, -3, -3 };
	array<float, 3> y = { -3.5, 5.2, -2 };
	array<float, 3> z = { 0, 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, Runner::result(z))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(3), "z"),
			Expression::operation(location, makeType<float>(3), OperationKind::subtract,
														Expression::reference(location, makeType<int>(3), "x"),
														Expression::reference(location, makeType<float>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 2, -3, -3 };
	array<float, 3> y = { -3.5, 5.2, -2 };
	array<float, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<float, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<float, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, Runner::result(zPtr))));

	for (const auto& [x, y, z] : llvm::zip(xPtr, yPtr, zPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member tMember(location, "t", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "t"),
			Expression::operation(location, makeType<int>(), OperationKind::subtract,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y"),
														Expression::reference(location, makeType<int>(), "z")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember, tMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 4, -7, -15 };
	array<int, 3> t = { 0, 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z, t] : llvm::zip(x, y, z, t))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, z, Runner::result(t))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::multiply,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 2> x = { 2, 5 };
	array<int, 2> y = { 3, -3 };
	array<int, 2> z = { 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, Runner::result(z))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::multiply,
														Expression::reference(location, makeType<float>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<float, 2> x = { 2.3, 5.7 };
	array<float, 2> y = { 23.57, -23.57 };
	array<float, 2> z = { 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, Runner::result(z))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::multiply,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 2, -3, -3 };
	array<float, 3> y = { -3.5, 5.2, -2 };
	array<float, 3> z = { 0, 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, Runner::result(z))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member tMember(location, "t", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "t"),
			Expression::operation(location, makeType<int>(), OperationKind::multiply,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y"),
														Expression::reference(location, makeType<int>(), "z")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember, tMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 10, 23, 57 };
	array<int, 3> y = { 10, 57, -23 };
	array<int, 3> z = { 4, -7, -15 };
	array<int, 3> t = { 0, 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z, t] : llvm::zip(x, y, z, t))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, z, Runner::result(t))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::multiply,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	int x = 2;
	array<int, 3> y = { 3, -5, 0 };
	array<int, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, yPtr, Runner::result(zPtr))));

	for (const auto& [y, z] : llvm::zip(yPtr, zPtr))
		EXPECT_EQ(z, x * y);
}

TEST(MathOps, mulIntegerScalarAndIntegerDynamicArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer[-1] y;
	 *   output Integer[-1] z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "z"),
			Expression::operation(location, makeType<int>(-1), OperationKind::multiply,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(-1), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	int x = 2;
	array<int, 3> y = { 3, -5, 0 };
	array<int, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, yPtr, Runner::result(zPtr))));

	for (const auto& [y, z] : llvm::zip(yPtr, zPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::multiply,
														Expression::reference(location, makeType<int>(3), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 3, -5, 0 };
	int y = 2;
	array<int, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, y, Runner::result(zPtr))));

	for (const auto& [x, z] : llvm::zip(xPtr, zPtr))
		EXPECT_EQ(z, x * y);
}

TEST(MathOps, mulIntegerDynamicArrayAndIntegerScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[-1] x;
	 *   input Integer y;
	 *   output Integer[-1] z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "z"),
			Expression::operation(location, makeType<int>(-1), OperationKind::multiply,
														Expression::reference(location, makeType<int>(-1), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 3, -5, 0 };
	int y = 2;
	array<int, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, y, Runner::result(zPtr))));

	for (const auto& [x, z] : llvm::zip(xPtr, zPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::multiply,
														Expression::reference(location, makeType<int>(3), "x"),
														Expression::reference(location, makeType<int>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 3, 5, 2 };
	array<int, 3> y = { 7, -2, 3 };
	int z = 0;

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, Runner::result(z))));

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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(4), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(4, 3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::multiply,
														Expression::reference(location, makeType<int>(4), "x"),
														Expression::reference(location, makeType<int>(4, 3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 4> x = { 1, 2, 3, 4 };
	array<int, 12> y = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	array<int, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 4 });
	ArrayDescriptor<int, 2> yPtr(y.data(), { 4, 3 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, Runner::result(zPtr))));

	EXPECT_EQ(zPtr.getSize(0), 3);

	EXPECT_EQ(zPtr[0], 70);
	EXPECT_EQ(zPtr[1], 80);
	EXPECT_EQ(zPtr[2], 90);
}

TEST(MathOps, mulIntegerStaticMatrixAndIntegerStaticVector)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[4, 3] x;
	 *   input Integer[3] y;
	 *   output Integer[4] z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(4, 3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(4), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(4), "z"),
			Expression::operation(location, makeType<int>(4), OperationKind::multiply,
														Expression::reference(location, makeType<int>(4, 3), "x"),
														Expression::reference(location, makeType<int>(3), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 12> x = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	array<int, 3> y = { 1, 2, 3 };
	array<int, 4> z = { 0, 0, 0, 0 };

	ArrayDescriptor<int, 2> xPtr(x.data(), { 4, 3 });
	ArrayDescriptor<int, 1> yPtr(y.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 4 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, Runner::result(zPtr))));

	EXPECT_EQ(zPtr.getSize(0), 4);

	EXPECT_EQ(zPtr[0], 14);
	EXPECT_EQ(zPtr[1], 32);
	EXPECT_EQ(zPtr[2], 50);
	EXPECT_EQ(zPtr[3], 68);
}

TEST(MathOps, mulIntegerStaticMatrixes)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[2, 3] x;
	 *   input Integer[3, 2] y;
	 *   output Integer[2, 2] z;
	 *
	 *   algorithm
	 *     z := x * y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(2, 3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(3, 2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(2, 2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(2, 2), "z"),
			Expression::operation(location, makeType<int>(2, 2), OperationKind::multiply,
														Expression::reference(location, makeType<int>(2, 3), "x"),
														Expression::reference(location, makeType<int>(3, 2), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 6> x = { 1, 2, 3, 4, 5, 6 };
	array<int, 6> y = { 1, 2, 3, 4, 5, 6 };
	array<int, 4> z = { 0, 0, 0, 0 };

	ArrayDescriptor<int, 2> xPtr(x.data(), { 2, 3 });
	ArrayDescriptor<int, 2> yPtr(y.data(), { 3, 2 });
	ArrayDescriptor<int, 2> zPtr(z.data(), { 2, 2 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, yPtr, Runner::result(zPtr))));

	EXPECT_EQ(zPtr.getSize(0), 2);
	EXPECT_EQ(zPtr.getSize(1), 2);

	EXPECT_EQ(zPtr.get(0, 0), 22);
	EXPECT_EQ(zPtr.get(0, 1), 28);
	EXPECT_EQ(zPtr.get(1, 0), 49);
	EXPECT_EQ(zPtr.get(1, 1), 64);
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::divide,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 2> x = { 6, 10 };
	array<int, 2> y = { 3, -5 };
	array<int, 2> z = { 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, Runner::result(z))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::divide,
														Expression::reference(location, makeType<float>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<float, 2> x = { 10.8, 10 };
	array<float, 2> y = { 3.6, -3.2 };
	array<float, 2> z = { 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, Runner::result(z))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member tMember(location, "t", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "t"),
			Expression::operation(location, makeType<int>(), OperationKind::divide,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y"),
														Expression::reference(location, makeType<int>(), "z")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember, tMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 120, 50, 0 };
	array<int, 3> y = { 2, 5, 5 };
	array<int, 3> z = { -3, 2, 2 };
	array<int, 3> t = { 0, 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z, t] : llvm::zip(x, y, z, t))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, z, Runner::result(t))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "z"),
			Expression::operation(location, makeType<float>(), OperationKind::divide,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<float>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 23, 10, -3 };
	array<float, 3> y = { -3.5, 3.2, -2 };
	array<float, 3> z = { 0, 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, Runner::result(z))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(3), "z"),
			Expression::operation(location, makeType<int>(3), OperationKind::divide,
														Expression::reference(location, makeType<int>(3), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 10, -5, 0 };
	int y = 2;
	array<int, 3> z = { 0, 0, 0 };

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(z.data(), { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, y, Runner::result(zPtr))));

	for (const auto& [x, z] : llvm::zip(xPtr, zPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1), "z"),
			Expression::operation(location, makeType<int>(-1), OperationKind::divide,
														Expression::reference(location, makeType<int>(-1), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 3> x = { 3, -5, 0 };
	int y = 2;

	ArrayDescriptor<int, 1> xPtr(x.data(), { 3 });
	ArrayDescriptor<int, 1> zPtr(nullptr, { 3 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, y, Runner::result(zPtr))));

	for (const auto& [x, z] : llvm::zip(xPtr, zPtr))
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::operation(location, makeType<int>(), OperationKind::powerOf,
														Expression::reference(location, makeType<int>(), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 2> x = { 3, 2 };
	array<int, 2> y = { 4, 0 };
	array<int, 2> z = { 0, 0 };

	Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, Runner::result(z))));
		EXPECT_EQ(z, pow(x, y));
	}
}

TEST(MathOps, powSquareMatrix)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[2, 2] x;
	 *   input Integer y;
	 *   output Integer[2, 2] z;
	 *
	 *   algorithm
	 *     z := x ^ y;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(2, 2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(2, 2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(2, 2), "z"),
			Expression::operation(location, makeType<int>(2, 2), OperationKind::powerOf,
														Expression::reference(location, makeType<int>(2, 2), "x"),
														Expression::reference(location, makeType<int>(), "y")));

	ClassContainer cls(Function(location, "main", true,
															{ xMember, yMember, zMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaConversionOptions conversionOptions;
	conversionOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, conversionOptions)));

	array<int, 4> x = { 1, 2, 3, 4 };
	int y = 3;
	array<int, 4> z = { 0, 0, 0, 0 };

	ArrayDescriptor<int, 2> xPtr(x.data(), { 2, 2 });
	ArrayDescriptor<int, 2> zPtr(z.data(), { 2, 2 });

	Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, y, Runner::result(zPtr))));

	EXPECT_EQ(zPtr.get(0, 0), 37);
	EXPECT_EQ(zPtr.get(0, 1), 54);
	EXPECT_EQ(zPtr.get(1, 0), 81);
	EXPECT_EQ(zPtr.get(1, 1), 118);
}
