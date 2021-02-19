#include <gtest/gtest.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/Dialect.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/frontend/Parser.hpp>
#include <modelica/mlirlowerer/MlirLowerer.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/Runner.h>
#include <modelica/utils/SourceRange.hpp>

using namespace modelica;
using namespace std;

TEST(FunctionLowerTest, test)	 // NOLINT
{
	llvm::DebugFlag = true;

	SourcePosition location = SourcePosition::unknown();

	Member xMember(location, "x", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member yMember(location, "y", makeType<BuiltInType::Boolean>(), TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Statement assignment = AssignmentStatement(
			location,
			Expression::reference(location, makeType<BuiltInType::Integer>(), "x"),
			Expression::constant(location, makeType<BuiltInType::Integer>(), 57));

	ClassContainer cls(Function(location, "main", true, { xMember, yMember }, Algorithm(location, assignment)));

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower(cls);

	module.dump();

	if (failed(convertToLLVMDialect(&context, module)))
		FAIL();

	module.dump();

	mlir::registerLLVMDialectTranslation(*module->getContext());
	llvm::LLVMContext llvmContext;
	auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
	llvmModule->print(llvm::errs(), nullptr);

	bool x = true;
	bool y = true;

	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	// Create the engine to run the code
	llvm::SmallVector<llvm::StringRef, 3> libraries;
	libraries.push_back("/opt/llvm/lib/libmlir_runner_utils.so");
	libraries.push_back("/opt/llvm/lib/libmlir_c_runner_utils.so");
	//libraries.push_back("/mnt/d/modelica/cmake-build-gcc-debug/lib/runtime/libruntime-d.so");

	//mlir::registerLLVMDialectTranslation(*module->getContext());

	auto maybeEngine = mlir::ExecutionEngine::create(module, nullptr, {}, llvm::None, libraries);

	if (!maybeEngine)
		llvm::errs() << "Failed to create the engine\n";

	auto& engine = maybeEngine.get();

	llvm::SmallVector<void*, 3> args;
	args.push_back((void*) &x);
	args.push_back((void*) &y);

	if (engine->invoke("main", x, mlir::ExecutionEngine::result(y)))
		llvm::errs() << "JIT invocation failed\n";

	//Runner runner(&context, module);
	//runner.run("main", x, y);

	EXPECT_EQ(y, false);


	/*
	SourcePosition location("-", 0, 0);

	Member x(location, "x", Type::Float(),
					 TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member y(location, "y", Type::Float(),
					 TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Member z(location, "z", Type::Float(),
					 TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Member t(location, "t", Type(BuiltInType::Float, { 2 }),
					 TypePrefix(ParameterQualifier::none, IOQualifier::input));

	Algorithm algorithm(location, {
			AssignmentStatement(Expression(location, Type::Float(), ReferenceAccess("y")),
													Expression(location, Type::Float(), Constant(23.0))),
			AssignmentStatement(Expression(location, Type::Float(), ReferenceAccess("z")),
													Expression(location, Type::Float(), ReferenceAccess("y"))),
			IfStatement(llvm::ArrayRef({ ConditionalBlock<Statement>(
					Expression::op<OperationKind::greaterEqual>(location, makeType<bool>(), Expression(location, Type::Float(), ReferenceAccess("y")), Expression(location, Type::Float(), ReferenceAccess("x"))),
					{
							AssignmentStatement(Expression(location, Type::Float(), ReferenceAccess("z")),
																	Expression(location, Type::Float(), Constant(57.0)))
					}
					),
					ConditionalBlock<Statement>(Expression::trueExp(location), {AssignmentStatement(Expression(location, Type::Float(), ReferenceAccess("z")),
																																													Expression(location, Type::Float(), Constant(44.0)))} )
			}))
			//AssignmentStatement(Expression(location, Type::Float(), ReferenceAccess("z")),
			//										Expression(location, Type::Float(), Call(Expression(Type::Float(), ReferenceAccess("test")))))
			//AssignmentStatement(Expression(location, Type::Float(), ReferenceAccess("z")),
			//										Expression(location, Type::Float(), OperationKind::add, Expression(Type::Float(), ReferenceAccess("x")), Expression(Type::Float(), ReferenceAccess("z")), Expression(Type::Float(), ReferenceAccess("z")), Expression(Type::Float(), ReferenceAccess("z")))),
			//AssignmentStatement(Expression(location, Type::Float(), ReferenceAccess("z")),
			//										Expression(location, Type::Float(), OperationKind::add, Expression(Type::Float(), ReferenceAccess("z")), Expression(Type::Float(), ReferenceAccess("z"))))
	});

	Function function(SourcePosition("-", 0, 0),
										"Foo", true, {x, y, z, t}, { algorithm });

	ClassContainer cls(function);

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	mlir::ModuleOp module = lowerer.lower({ cls });
	module.dump();

	if (failed(convertToLLVMDialect(&context, module)))
		return;

	llvm::LLVMContext llvmContext;
	auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
	llvmModule->print(llvm::errs(), nullptr);

	if (!llvmModule) {
		llvm::errs() << "Failed to emit LLVM IR\n";
	}
	else
	{
		// Initialize LLVM targets.
		//mlir::llvm::InitializeNativeTarget();
		//mlir::llvm::InitializeNativeTargetAsmPrinter();
		//mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

		/// Optionally run an optimization pipeline over the llvm module.
		auto optPipeline = mlir::makeOptimizingTransformer(
				3, // optLevel
				0, // sizeLevel
				nullptr); // targetMachine

		if (auto err = optPipeline(llvmModule.get())) {
			llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
		}

		llvm::errs() << *llvmModule << "\n";
	}
	 */
}
