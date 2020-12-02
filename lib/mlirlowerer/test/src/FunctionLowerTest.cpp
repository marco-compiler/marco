#include <gtest/gtest.h>
#include <modelica/frontend/Parser.hpp>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
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

	Algorithm algorithm({
			AssignmentStatement(Expression(Type::Float(), ReferenceAccess("y")),
													Expression(Type::Float(), ReferenceAccess("x")))
	});

	Function function(SourcePosition("-", 0, 0),
										"Foo", true, {x, y}, { algorithm });

	mlir::MLIRContext context;
	MlirLowerer lowerer(context);
	lowerer.lower(function).dump();
}
