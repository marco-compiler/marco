#include "gtest/gtest.h"

/*
#include "marco/AST/Parser.h"

using namespace marco;
using namespace marco::ast;

TEST(Parser, inlinableFunction)	 // NOLINT
{
	Parser parser("function foo"
								"  algorithm"
								"  annotation(inline = true);"
								"end foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	EXPECT_TRUE((*ast)->get<StandardFunction>()->getAnnotation()->getInlineProperty());
}
*/