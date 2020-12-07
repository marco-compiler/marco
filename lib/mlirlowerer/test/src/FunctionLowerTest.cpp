#include <gtest/gtest.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/IR/Dialect.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/frontend/Parser.hpp>
#include <modelica/mlirlowerer/LLVMLoweringPass.hpp>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/mlirlowerer/ModelicaDialect.hpp>
#include <modelica/utils/SourceRange.hpp>

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

	Member x("x", Type::Float(),
					 TypePrefix(ParameterQualifier::none, IOQualifier::input));
	Member y("y", Type::Float(),
					 TypePrefix(ParameterQualifier::none, IOQualifier::output));
	Member z("z", Type::Float(),
					 TypePrefix(ParameterQualifier::none, IOQualifier::output));

	Algorithm algorithm({
			AssignmentStatement(Expression(Type::Float(), ReferenceAccess("y")),
													Expression(Type::Float(), Constant(23.0))),
			AssignmentStatement(Expression(Type::Float(), ReferenceAccess("z")),
													Expression(Type::Float(), ReferenceAccess("y")))
			//AssignmentStatement(Expression(Type::Float(), ReferenceAccess("z")),
			//										Expression(Type::Float(), Call(Expression(Type::Float(), ReferenceAccess("test")))))
			//AssignmentStatement(Expression(Type::Float(), ReferenceAccess("z")),
			//										Expression(Type::Float(), OperationKind::add, Expression(Type::Float(), ReferenceAccess("x")), Expression(Type::Float(), ReferenceAccess("z")), Expression(Type::Float(), ReferenceAccess("z")), Expression(Type::Float(), ReferenceAccess("z")))),
			//AssignmentStatement(Expression(Type::Float(), ReferenceAccess("z")),
			//										Expression(Type::Float(), OperationKind::add, Expression(Type::Float(), ReferenceAccess("z")), Expression(Type::Float(), ReferenceAccess("z"))))
	});

	Function function(SourcePosition("-", 0, 0),
										"Foo", true, {x, y, z}, { algorithm });

	mlir::registerDialect<ModelicaDialect>();
	mlir::registerDialect<mlir::StandardOpsDialect>();
	mlir::registerDialect<mlir::LLVM::LLVMDialect>();

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);

	mlir::ModuleOp module = mlir::ModuleOp::create(lowerer.builder.getUnknownLoc());
	auto lowered = lowerer.lower(function);
	module.push_back(lowered);

	module.dump();

	mlir::PassManager pm(&context);
	pm.addPass(std::make_unique<LLVMLoweringPass>());
	pm.run(module);

	//mlir::translateModuleToLLVMIR(module);
	module.dump();
}
