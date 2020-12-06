#include <gtest/gtest.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/IR/Dialect.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/frontend/Parser.hpp>
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
													Expression(Type::Float(), Constant(57.0)))
	});

	Function function(SourcePosition("-", 0, 0),
										"Foo", true, {x, y, z}, { algorithm });

	mlir::registerDialect<ModelicaDialect>();
	mlir::registerDialect<mlir::StandardOpsDialect>();

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	auto lowered = lowerer.lower(function);
	//lowered.dump();

	mlir::ConversionTarget target(context);
	target.addLegalDialect<mlir::LLVM::LLVMDialect>();
	//mlir::LLVMTypeConverter typeConverter(context);

	//mlir::OwningRewritePatternList patterns;
	//mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);

	//mlir::ModuleOp module = mlir::ModuleOp::create(lowerer.builder.getUnknownLoc());
	//module.push_back(lowered);

	//mlir::applyFullConversion(lowered, target, patterns);
	lowered.dump();
}
