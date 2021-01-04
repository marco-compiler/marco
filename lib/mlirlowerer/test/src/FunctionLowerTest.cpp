#include <gtest/gtest.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/IR/Dialect.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/frontend/Parser.hpp>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/mlirlowerer/ModelicaDialect.hpp>
#include <modelica/utils/SourceRange.hpp>
#include <mlir/ExecutionEngine/OptUtils.h>

using namespace modelica;
using namespace std;

TEST(FunctionLowerTest, test)	 // NOLINT
{
	/**
	 * function Foo
	 *   input Real x;
	 *   output Real y;
	 * algorithm
	 *   y := x;
	 * end Foo
	 */

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
