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

	SourcePosition location("-", 0, 0);

	Algorithm algorithm({
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
										"Foo", true, {x, y, z}, { algorithm });

	mlir::registerDialect<ModelicaDialect>();
	mlir::registerDialect<mlir::StandardOpsDialect>();
	mlir::registerDialect<mlir::scf::SCFDialect>();
	mlir::registerDialect<mlir::LLVM::LLVMDialect>();

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	auto lowered = lowerer.lower(function);
	//lowered.dump();

	mlir::ModuleOp module = mlir::ModuleOp::create(lowerer.builder.getUnknownLoc());
	module.push_back(lowered);

	module.dump();

	mlir::PassManager pm(&context);
	pm.addPass(std::make_unique<LLVMLoweringPass>());
	pm.run(module);
	module.dump();

	auto result = mlir::translateModuleToLLVMIR(module);
	//result->dump(); // doesn't work
	result->print(llvm::errs(), nullptr);

}
