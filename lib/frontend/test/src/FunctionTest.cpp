#include <gtest/gtest.h>
#include <llvm/Support/Error.h>
#include <modelica/frontend/Parser.hpp>
#include <modelica/frontend/TypeChecker.hpp>
#include <modelica/utils/ErrorTest.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

TEST(FunctionParse, functionName)	 // NOLINT
{
	Parser parser("function Foo end Foo;");
	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);
	ASSERT_EQ("Foo", ast.getName());
}

TEST(FunctionParse, singleAlgorithm)	// NOLINT
{
	Parser parser("function Foo"
								"	algorithm x := 0;"
								"end Foo;");

	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);
	ASSERT_EQ(1, ast.getAlgorithms().size());
}

TEST(FunctionParse, multipleAlgorithms)	 // NOLINT
{
	Parser parser("function Foo"
								"	algorithm x := 0;"
								"	algorithm y := 0;"
								"end Foo;");

	auto expectedAst = parser.classDefinition();

	if (!expectedAst)
		FAIL();

	auto ast = move(*expectedAst);
	ASSERT_EQ(2, ast.getAlgorithms().size());
}

TEST(FunctionTypeCheck, publicMembersMustBeInputOrOutput)
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
	ASSERT_ERROR(
			typeChecker.checkType<ClassType::Function>(ast, {}), BadSemantic);
}

TEST(FunctionTypeCheck, assignmentToInputMember)
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
	ASSERT_ERROR(
			typeChecker.checkType<ClassType::Function>(ast, {}), BadSemantic);
}

TEST(FunctionTypeCheck, assignmentToInputArray)
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
	ASSERT_ERROR(
			typeChecker.checkType<ClassType::Function>(ast, {}), BadSemantic);
}

TEST(FunctionTypeCheck, directDerCall)
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
	ASSERT_ERROR(
			typeChecker.checkType<ClassType::Function>(ast, {}), BadSemantic);
}

TEST(FunctionTypeCheck, derInsideParameters)
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
	ASSERT_ERROR(
			typeChecker.checkType<ClassType::Function>(ast, {}), BadSemantic);
}
