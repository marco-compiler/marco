#include "gtest/gtest.h"
#include "marco/ast/Parser.h"

using namespace marco;
using namespace marco::ast;
using namespace std;

TEST(RecordTest, empty)	 // NOLINT
{
	/*
	Parser parser("record Vector end Vector;");

	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);

	EXPECT_TRUE(ast.isA<ClassType::Record>());
	ASSERT_EQ(0, ast.getMembers().size());
	 */
}

TEST(RecordTest, vector)	// NOLINT
{
	/*
	Parser parser("record Vector \"A vector in 3D space\""
								"	Real x;"
								"	Real y;"
								"	Real z;"
								"	end Vector;");

	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);

	EXPECT_TRUE(ast.isA<ClassType::Record>());

	auto& members = ast.getMembers();
	ASSERT_EQ(3, members.size());
	EXPECT_EQ("x", members[0].getName());
	EXPECT_EQ("y", members[1].getName());
	EXPECT_EQ("z", members[2].getName());
	 */
}
