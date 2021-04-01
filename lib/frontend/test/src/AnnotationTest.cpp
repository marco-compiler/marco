#include <gtest/gtest.h>
#include <modelica/frontend/Parser.hpp>

using namespace modelica;
using namespace frontend;

TEST(Parser, inlineAnnotationTrue)	 // NOLINT
{
	Parser parser("annotation(Inline = true)");

	auto ast = parser.annotation();
	ASSERT_FALSE(!ast);

	EXPECT_TRUE(ast->getInlineProperty());
}

TEST(Parser, inlineAnnotationFalse)	 // NOLINT
{
	Parser parser("annotation(Inline = false)");

	auto ast = parser.annotation();
	ASSERT_FALSE(!ast);

	EXPECT_FALSE(ast->getInlineProperty());
}

TEST(Parser, inlinableFunction)	 // NOLINT
{
	Parser parser("function foo"
								"  algorithm"
								"  annotation(Inline = true);"
								"end foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	EXPECT_TRUE(ast->get<Function>().getAnnotation().getInlineProperty());
}
