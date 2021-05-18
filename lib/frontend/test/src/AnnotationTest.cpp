#include <gtest/gtest.h>
#include <modelica/frontend/Parser.hpp>

using namespace modelica;
using namespace frontend;

TEST(Parser, inlineAnnotationTrue)	 // NOLINT
{
	Parser parser("annotation(inline = true)");

	auto ast = parser.annotation();
	ASSERT_FALSE(!ast);

	EXPECT_TRUE((*ast)->getInlineProperty());
}

TEST(Parser, inlineAnnotationFalse)	 // NOLINT
{
	Parser parser("annotation(inline = false)");

	auto ast = parser.annotation();
	ASSERT_FALSE(!ast);

	EXPECT_FALSE((*ast)->getInlineProperty());
}

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

TEST(Parser, inverseFunctionAnnotation)	 // NOLINT
{
	Parser parser("annotation(inverse(y = foo1(x, z), z = foo2(x, y)))");

	auto ast = parser.annotation();
	ASSERT_FALSE(!ast);

	auto annotation = (*ast)->getInverseFunctionAnnotation();

	EXPECT_TRUE(annotation.isInvertible("y"));
	EXPECT_EQ(annotation.getInverseFunction("y"), "foo1");
	EXPECT_EQ(annotation.getInverseArgs("y").size(), 2);
	EXPECT_EQ(annotation.getInverseArgs("y")[0], "x");
	EXPECT_EQ(annotation.getInverseArgs("y")[1], "z");

	EXPECT_TRUE(annotation.isInvertible("z"));
	EXPECT_EQ(annotation.getInverseFunction("z"), "foo2");
	EXPECT_EQ(annotation.getInverseArgs("z").size(), 2);
	EXPECT_EQ(annotation.getInverseArgs("z")[0], "x");
	EXPECT_EQ(annotation.getInverseArgs("z")[1], "y");
}
