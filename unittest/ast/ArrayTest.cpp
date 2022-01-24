#include "gtest/gtest.h"
#include "llvm/Support/Error.h"
#include "marco/ast/Parser.h"
#include "marco/ast/Passes.h"
#include "marco/utils/ErrorTest.h"

using namespace marco::ast;
using namespace std;

TEST(TypeChecking, arrayOfIntegers)	// NOLINT
{
	Parser parser("{ 1, 2, 3 }");

	auto ast = parser.expression();
	ASSERT_FALSE(!ast);

	TypeChecker typeChecker;
	EXPECT_FALSE(typeChecker.run<Expression>(**ast));

	EXPECT_EQ((*ast)->getType(), makeType<int>(3));
}

TEST(TypeChecking, arrayOfReals)	// NOLINT
{
	Parser parser("{ 1.0, 2, 3.0 }");

	auto ast = parser.expression();
	ASSERT_FALSE(!ast);

	TypeChecker typeChecker;
	EXPECT_FALSE(typeChecker.run<Expression>(**ast));

	EXPECT_EQ((*ast)->getType(), makeType<float>(3));
}
