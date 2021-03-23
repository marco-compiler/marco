#include <gtest/gtest.h>
#include <llvm/Support/Error.h>
#include <modelica/frontend/Parser.hpp>
#include <modelica/frontend/Passes.h>
#include <modelica/utils/ErrorTest.hpp>

using namespace modelica;

TEST(Parser, functionName)	 // NOLINT
{
	Parser parser("function Foo end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	EXPECT_EQ(ast->get<Function>().getName(), "Foo");
}

TEST(Parser, functionAllMembers)	// NOLINT
{
	Parser parser("function Foo"
								"	 input Integer x;"
								"	 input Integer y;"
								"  output Real z;"
								"protected"
								"  Integer t;"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	const auto& members = ast->get<Function>().getMembers();
	EXPECT_EQ(members.size(), 4);
	EXPECT_EQ(members[0]->getName(), "x");
	EXPECT_EQ(members[1]->getName(), "y");
	EXPECT_EQ(members[2]->getName(), "z");
	EXPECT_EQ(members[3]->getName(), "t");
}

TEST(Parser, functionInputMembers)	// NOLINT
{
	Parser parser("function Foo"
								"	 input Integer x;"
								"	 input Integer y;"
								"  output Real z;"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	const auto& members = ast->get<Function>().getArgs();
	EXPECT_EQ(members.size(), 2);
	EXPECT_EQ(members[0]->getName(), "x");
	EXPECT_EQ(members[1]->getName(), "y");
}

TEST(Parser, functionOutputMembers)	// NOLINT
{
	Parser parser("function Foo"
								"	 input Integer x;"
								"  output Real y;"
								"  output Real z;"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	const auto& members = ast->get<Function>().getResults();
	EXPECT_EQ(members.size(), 2);
	EXPECT_EQ(members[0]->getName(), "y");
	EXPECT_EQ(members[1]->getName(), "z");
}

TEST(Parser, functionProtectedMembers)	// NOLINT
{
	Parser parser("function Foo"
								"	 input Integer x;"
								"  output Real y;"
								"protected"
								"  Integer z;"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	const auto& members = ast->get<Function>().getProtectedMembers();
	EXPECT_EQ(members.size(), 1);
	EXPECT_EQ(members[0]->getName(), "z");
}

TEST(Parser, functionAlgorithm)	// NOLINT
{
	Parser parser("function Foo"
								"	 algorithm"
								"    x := 0;"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	EXPECT_EQ(ast->get<Function>().getAlgorithms().size(), 1);
}

/*
TEST(FunctionTypeCheck, publicMembersMustBeInputOrOutput)	 // NOLINT
{
	Parser parser("function Foo"
								"	input Real x;"
								"	output Real y;"
								" Real z;"
								"	algorithm y := der(x);"
								"end Foo;");

	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);
	TypeChecker typeChecker;
	EXPECT_ERROR(typeChecker.check(ast), BadSemantic);
}

TEST(FunctionTypeCheck, assignmentToInputMember)	// NOLINT
{
	Parser parser("function Foo"
								"	input Real x;"
								"	output Real y;"
								"	algorithm"
								"		x := x^2;"
								"		y := x;"
								"end Foo;");

	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);
	TypeChecker typeChecker;
	EXPECT_ERROR(typeChecker.check(ast), BadSemantic);
}

TEST(FunctionTypeCheck, assignmentToInputArray)	 // NOLINT
{
	Parser parser("function Foo"
								"	input Real[2] x;"
								"	output Real y;"
								"	algorithm"
								"		x[1] := x[2]^2;"
								"		y := x[1] + x[2];"
								"end Foo;");

	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);
	TypeChecker typeChecker;
	EXPECT_ERROR(typeChecker.check(ast), BadSemantic);
}

TEST(FunctionTypeCheck, directDerCall)	// NOLINT
{
	Parser parser("function Foo"
								"	input Real x;"
								"	output Real y;"
								"	algorithm y := der(x);"
								"end Foo;");

	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);
	TypeChecker typeChecker;
	EXPECT_ERROR(typeChecker.check(ast), BadSemantic);
}

TEST(FunctionTypeCheck, derInsideParameters)	// NOLINT
{
	Parser parser("function Foo"
								"	input Real x;"
								" input Real y;"
								"	output Real z;"
								"	algorithm y := x + Foo(der(x));"
								"end Foo;");

	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);
	TypeChecker typeChecker;
	EXPECT_ERROR(typeChecker.check(ast), BadSemantic);
}
*/