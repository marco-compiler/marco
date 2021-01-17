#include <gtest/gtest.h>
#include <mlir/IR/Dialect.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/mlirlowerer/Runner.hpp>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
using namespace std;

TEST(Input, integerScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer x;
	 *   output Integer y;
	 *   algorithm
	 *     y := x;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
			Expression::reference(location, makeType<BuiltInType::Integer>(), "x"));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);
	int x = 57;
	int y = 0;
	runner.run("main", x, y);
	EXPECT_EQ(y, 57);
}

TEST(Input, floatScalar)	 // NOLINT
{
	/**
	 * function main
	 *   input Real x;
	 *   output Real y;
	 *   algorithm
	 *     y := x;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Float>(), "y"),
			Expression::reference(location, makeType<BuiltInType::Float>(), "x"));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);
	float x = 57;
	float y = 0;
	runner.run("main", x, y);
	EXPECT_EQ(y, 57.0);
}

TEST(Input, integerArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Integer[2] x;
	 *   output Integer y;
	 *   algorithm
	 *     y := x[0] + x[1];
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Integer>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "y"),
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::add,
														Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
																									Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
														Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
																									Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 1))));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	if (failed(modelica::convertToLLVMDialect(&context, module)))
		llvm::errs() << "Failed to convert to LLVM dialect\n";

	llvm::LLVMContext llvmContext;
	auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);

	llvm::errs() << *llvmModule;

	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();
	auto maybeEngine = mlir::ExecutionEngine::create(module);

	if (!maybeEngine)
		llvm::errs() << "Failed to create the engine\n";

	auto& engine = maybeEngine.get();

	int x[2] = {23, 57};
	int* bufferPtr = x;
	int* alignedPtr = x;
	long offset = 0;
	long size = 2;
	long stride = 1;
	int y;

	StridedMemRefType<int, 2> arr{x, x, 0, {2}, {1}};
	llvm::SmallVector<void*, 3> args;
	args.push_back((void*) &bufferPtr);
	args.push_back((void*) &alignedPtr);
	args.push_back((void*) &offset);
	args.push_back((void*) &size);
	args.push_back((void*) &stride);
	args.push_back((void*) &y);

	if (engine->invoke("main", args))
		llvm::errs() << "JIT invocation failed\n";

	EXPECT_EQ(y, 80);
}

TEST(Input, floatArray)	 // NOLINT
{
	/**
	 * function main
	 *   input Real[2] x;
	 *   output Real y;
	 *   algorithm
	 *     y := x[0] + x[1];
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Float>(2), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Float>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Float>(), "y"),
			Expression::operation(location, makeType<BuiltInType::Float>(), OperationKind::add,
														Expression::operation(location, makeType<BuiltInType::Float>(), OperationKind::subscription,
																									Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
														Expression::operation(location, makeType<BuiltInType::Float>(), OperationKind::subscription,
																									Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
																									Expression::constant(location, makeType<BuiltInType::Integer>(), 1))));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);
	array<float, 2> xArray = { 23.0, 57.0 };
	StridedMemRefType<float, 2> xMemRef{xArray.data(), xArray.data(), 0, {2}, {1}};
	auto* x = &xMemRef;
	float y = 0;

	runner.run("_mlir_ciface_main", x, y);

	EXPECT_FLOAT_EQ(y, 80);
}

TEST(Output, integerArray)	 // NOLINT
{
	/**
	 * function main
	 *   output Integer[2] x;
	 *   algorithm
	 *     y[0] := 23;
	 *     y[1] := 57;
	 * end main
	 */

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Integer>(2), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment0 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
														Expression::constant(location, makeType<BuiltInType::Integer>(), 0)),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 23));

	Statement assignment1 = AssignmentStatement(
			location,
			Expression::operation(location, makeType<BuiltInType::Integer>(), OperationKind::subscription,
														Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
														Expression::constant(location, makeType<BuiltInType::Integer>(), 1)),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57));

	ClassContainer cls(Function(location, "main", true, xMember, Algorithm(location, { assignment0, assignment1 })));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context, false);
	mlir::ModuleOp module = lowerer.lower(cls);

	Runner runner(&context, module);
	array<int, 2> xArray = { 0, 0 };
	StridedMemRefType<int, 2> xMemRef{xArray.data(), xArray.data(), 0, {2}, {1}};
	auto* x = &xMemRef;

	runner.run("_mlir_ciface_main", x);

	EXPECT_EQ(xArray[0], 23);
	EXPECT_EQ(xArray[1], 57);
}
