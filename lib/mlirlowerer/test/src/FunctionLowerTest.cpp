#include <gtest/gtest.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Dialect.h>
#include <llvm/ADT/ScopedHashTable.h>
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
													Expression(Type::Float(), Constant(23))),
			AssignmentStatement(Expression(Type::Float(), ReferenceAccess("z")),
													Expression(Type::Float(), Constant(57)))
	});

	Function function(SourcePosition("-", 0, 0),
										"Foo", true, {x, y, z}, { algorithm });

	mlir::registerDialect<ModelicaDialect>();
	mlir::registerDialect<mlir::StandardOpsDialect>();

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	lowerer.lower(function).dump();
}
