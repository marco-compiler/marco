#include "gtest/gtest.h"

#include "marco/frontend/Parser.h"

using namespace marco;
using namespace frontend;

TEST(classTest, modelComment)	 // NOLINT
{
	Parser parser("model C \"comment!\" parameter Real A = 315.15;"
								"Real[10, 10, 4] T(start = A); end C;");

	auto expectedAST = parser.classDefinition();
	if (!expectedAST)
		FAIL();
	SUCCEED();
}
