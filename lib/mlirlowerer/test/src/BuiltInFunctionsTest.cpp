#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <modelica/frontend/AST.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/runtime/ArrayDescriptor.h>
#include <modelica/utils/SourcePosition.h>
#include <numeric>

using namespace modelica;
using namespace frontend;
using namespace codegen;
using namespace std;

TEST(BuiltInOps, ndims)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[3] x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := ndims(x);
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::call(location, makeType<int>(),
			    Expression::reference(location, makeType<int>(), "ndims"),
											 Expression::reference(location, makeType<int>(3), "x")));

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

	array<int, 3> x = { 10, 23, -57 };
	ArrayDescriptor<int, 1> xPtr(x);

	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, jit::Runner::result(y))));
	EXPECT_EQ(y, xPtr.getRank());
}

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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(3, 2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::call(location, makeType<int>(),
											 Expression::reference(location, makeType<int>(), "size"),
											 Expression::reference(location, makeType<int>(3, 2), "x"),
											 Expression::constant(location, makeType<int>(), 2)));

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

	array<int, 6> x = { 1, 2, 3, 4, 5, 6 };
	ArrayDescriptor<int, 2> xPtr(x.data(), { 3, 2 });

	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, jit::Runner::result(y))));
	EXPECT_EQ(y, xPtr.getDimensionSize(1));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(4, 3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(2), "y"),
			Expression::call(location, makeType<int>(2),
											 Expression::reference(location, makeType<int>(2), "size"),
											 Expression::reference(location, makeType<int>(4, 3), "x")));

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

	array<int, 12> x = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	ArrayDescriptor<int, 2> xPtr(x.data(), { 4, 3 });

	array<int, 2> y = { 0, 0 };
	ArrayDescriptor<int, 1> yPtr(y);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, jit::Runner::result(yPtr))));

	EXPECT_EQ(yPtr[0], xPtr.getDimensionSize(0));
	EXPECT_EQ(yPtr[1], xPtr.getDimensionSize(1));
}

TEST(BuiltInOps, identityMatrix)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer[:,:] y;
	 *
	 *   algorithm
	 *     y := identity(x);
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(-1, -1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1, -1), "y"),
			Expression::call(location, makeType<int>(-1, -1),
											 Expression::reference(location, makeType<int>(-1, -1), "identity"),
											 Expression::reference(location, makeType<int>(), "x")));

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

	int x = 3;

	array<int, 3> y = { 2, 2, 2 };
	ArrayDescriptor<int, 1> yPtr(y);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", x, jit::Runner::result(yPtr))));

	EXPECT_EQ(yPtr.getRank(), 2);

	for (long i = 0; i < 3; ++i)
		for (long j = 0; j < 3; ++j)
			EXPECT_EQ(yPtr.get(i, j), i == j ? 1 : 0);
}

TEST(BuiltInOps, diagonalMatrix)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[:] x;
	 *   output Integer[:,:] y;
	 *
	 *   algorithm
	 *     y := diagonal(x);
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(-1, -1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1, -1), "y"),
			Expression::call(location, makeType<int>(-1, -1),
											 Expression::reference(location, makeType<int>(-1, -1), "diagonal"),
											 Expression::reference(location, makeType<int>(-1), "x")));

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

	array<int, 3> x = { 1, 2, 3 };
	ArrayDescriptor<int, 1> xPtr(x);

	array<int, 3> y = { 2, 2, 2 };
	ArrayDescriptor<int, 1> yPtr(y);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, jit::Runner::result(yPtr))));

	EXPECT_EQ(yPtr.getRank(), 2);

	for (long i = 0; i < 3; ++i)
		for (long j = 0; j < 3; ++j)
			EXPECT_EQ(yPtr.get(i, j), i == j ? x[i] : 0);
}

TEST(BuiltInOps, zerosMatrix)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer n1;
	 *   input Integer n2;
	 *   output Integer[:,:] y;
	 *
	 *   algorithm
	 *     y := zeros(n1, n2);
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member n1Member(location, "n1", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member n2Member(location, "n2", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(-1, -1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1, -1), "y"),
			Expression::call(location, makeType<int>(-1, -1),
											 Expression::reference(location, makeType<int>(-1, -1), "zeros"),
											 Expression::reference(location, makeType<int>(), "n1"),
											 Expression::reference(location, makeType<int>(), "n2")));

	ClassContainer cls(Function(location, "main", true,
															{ n1Member, n2Member, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	const unsigned long n1 = 3;
	const unsigned long n2 = 2;

	array<int, 6> y = { 1, 1, 1, 1, 1, 1 };
	ArrayDescriptor<int, 2> yPtr(y.data(), { n1, n2 });

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", n1, n2, jit::Runner::result(yPtr))));

	EXPECT_EQ(yPtr.getRank(), 2);
	EXPECT_EQ(yPtr.getDimensionSize(0), n1);
	EXPECT_EQ(yPtr.getDimensionSize(1), n2);

	for (size_t i = 0; i < yPtr.getDimensionSize(0); ++i)
		for (size_t j = 0; j < yPtr.getDimensionSize(1); ++j)
			EXPECT_EQ(yPtr.get(i, j), 0);
}

TEST(BuiltInOps, onesMatrix)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer n1;
	 *   input Integer n2;
	 *   output Integer[:,:] y;
	 *
	 *   algorithm
	 *     y := ones(n1, n2);
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member n1Member(location, "n1", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member n2Member(location, "n2", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(-1, -1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(-1, -1), "y"),
			Expression::call(location, makeType<int>(-1, -1),
											 Expression::reference(location, makeType<int>(-1, -1), "ones"),
											 Expression::reference(location, makeType<int>(), "n1"),
											 Expression::reference(location, makeType<int>(), "n2")));

	ClassContainer cls(Function(location, "main", true,
															{ n1Member, n2Member, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	const unsigned long n1 = 3;
	const unsigned long n2 = 2;

	array<int, 6> y = { 0, 0, 0, 0, 0, 0 };
	ArrayDescriptor<int, 2> yPtr(y.data(), { n1, n2 });

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", n1, n2, jit::Runner::result(yPtr))));

	EXPECT_EQ(yPtr.getRank(), 2);
	EXPECT_EQ(yPtr.getDimensionSize(0), n1);
	EXPECT_EQ(yPtr.getDimensionSize(1), n2);

	for (size_t i = 0; i < yPtr.getDimensionSize(0); ++i)
		for (size_t j = 0; j < yPtr.getDimensionSize(1); ++j)
			EXPECT_EQ(yPtr.get(i, j), 1);
}

TEST(BuiltInOps, linspace)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer start;
	 *   input Integer end;
	 *   input Integer n;
	 *   output Real[:] y;
	 *
	 *   algorithm
	 *     y := linspace(start, end, n);
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member startMember(location, "start", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member endMember(location, "end", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member nMember(location, "n", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(-1), "y"),
			Expression::call(location, makeType<float>(-1),
											 Expression::reference(location, makeType<float>(-1), "linspace"),
											 Expression::reference(location, makeType<int>(), "start"),
											 Expression::reference(location, makeType<int>(), "end"),
											 Expression::reference(location, makeType<int>(), "n")));

	ClassContainer cls(Function(location, "main", true,
															{ startMember, endMember, nMember, yMember },
															Algorithm(location, assignment)));

	mlir::MLIRContext context;

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);

	auto module = lowerer.lower(cls);

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	int start = 0;
	int end = 1;
	const int n = 17;

	array<float, n> y = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	ArrayDescriptor<float, 1> yPtr(y);

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", start, end, n, jit::Runner::result(yPtr))));

	for (size_t i = 0; i < n; ++i)
		EXPECT_FLOAT_EQ(yPtr[i], start +  i * ((float) (end - start) / (n - 1)));
}

TEST(BuiltInOps, minArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[:] x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := min(x);
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::call(location, makeType<int>(),
											 Expression::reference(location, makeType<int>(), "min"),
											 Expression::reference(location, makeType<int>(), "x")));

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

	array<int, 5> x = { -1, 9, -3, 0, 4 };
	ArrayDescriptor<int, 1> xPtr(x);

	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, jit::Runner::result(y))));

	EXPECT_EQ(y, *std::min_element(x.begin(), x.end()));
}

TEST(BuiltInOps, minScalars)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Integer z;
	 *
	 *   algorithm
	 *     z := min(x, y);
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::call(location, makeType<int>(),
											 Expression::reference(location, makeType<int>(), "min"),
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

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 3> x = { 1, 2 };
	array<int, 3> y = { 2, 1 };
	array<int, 3> z = { 2, 2 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, std::min(x, y));
	}
}

TEST(BuiltInOps, maxArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[:] x;
	 *   output Integer y;
	 *
	 *   algorithm
	 *     y := max(x);
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(-1), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::call(location, makeType<int>(),
											 Expression::reference(location, makeType<int>(), "max"),
											 Expression::reference(location, makeType<int>(), "x")));

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

	array<int, 5> x = { -1, 9, -3, 0, 4 };
	ArrayDescriptor<int, 1> xPtr(x);

	int y = 0;

	jit::Runner runner(*module);
	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, jit::Runner::result(y))));

	EXPECT_EQ(y, *std::max_element(x.begin(), x.end()));
}

TEST(BuiltInOps, maxScalars)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   input Integer y;
	 *   output Integer z;
	 *
	 *   algorithm
	 *     z := max(x, y);
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member zMember(location, "z", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "z"),
			Expression::call(location, makeType<int>(),
											 Expression::reference(location, makeType<int>(), "max"),
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

	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	array<int, 3> x = { 1, 2 };
	array<int, 3> y = { 2, 1 };
	array<int, 3> z = { 1, 1 };

	jit::Runner runner(*module);

	for (const auto& [x, y, z] : llvm::zip(x, y, z))
	{
		ASSERT_TRUE(mlir::succeeded(runner.run("main", x, y, jit::Runner::result(z))));
		EXPECT_EQ(z, std::max(x, y));
	}
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

	ModelicaOptions modelicaOptions;
	modelicaOptions.x64 = false;
	MLIRLowerer lowerer(context, modelicaOptions);
	
	auto module = lowerer.lower(cls);
	
	ModelicaLoweringOptions loweringOptions;
	loweringOptions.llvmOptions.emitCWrappers = true;
	ASSERT_TRUE(module && !failed(lowerer.convertToLLVMDialect(*module, loweringOptions)));

	jit::Runner runner(*module);

	array<int, 3> x = { 1, 2, 3 };
	ArrayDescriptor<int, 1> xPtr(x);

	int y = 0;

	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, jit::Runner::result(y))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "y"),
			Expression::call(location, makeType<float>(),
											 Expression::reference(location, makeType<float>(), "sum"),
											 Expression::reference(location, makeType<float>(3), "x")));

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

	jit::Runner runner(*module);

	array<float, 3> x = { 1, 2, 3 };
	ArrayDescriptor<float, 1> xPtr(x);

	float y = 0;

	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, jit::Runner::result(y))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<int>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<int>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<int>(), "y"),
			Expression::call(location, makeType<int>(),
											 Expression::reference(location, makeType<int>(), "product"),
											 Expression::reference(location, makeType<int>(3), "x")));

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

	jit::Runner runner(*module);

	array<int, 3> x = { 1, 2, 3 };
	ArrayDescriptor<int, 1> xPtr(x);

	int y = 0;

	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, jit::Runner::result(y))));
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

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<float>(3), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<float>(), "y"),
			Expression::call(location, makeType<float>(),
											 Expression::reference(location, makeType<float>(), "product"),
											 Expression::reference(location, makeType<float>(3), "x")));

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

	jit::Runner runner(*module);

	array<float, 3> x = { 1, 2, 3 };
	ArrayDescriptor<float, 1> xPtr(x);

	float y = 0;

	ASSERT_TRUE(mlir::succeeded(runner.run("main", xPtr, jit::Runner::result(y))));
	EXPECT_EQ(y, std::accumulate(x.begin(), x.end(), 1, std::multiplies<>()));
}
