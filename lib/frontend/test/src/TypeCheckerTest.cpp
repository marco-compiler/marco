#include <gtest/gtest.h>
#include <llvm/Support/Error.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/Parser.hpp>
#include <modelica/frontend/Type.hpp>
#include <modelica/frontend/TypeChecker.hpp>
#include <modelica/utils/ErrorTest.hpp>

using namespace std;
using namespace modelica;
using namespace llvm;

TEST(TypeCheckTest, sumOfIntShouldProduceInt)	 // NOLINT
{
	Expression exp = Expression::op<OperationKind::add>(
			Type::unknown(),
			Expression(makeType<int>(), 0),
			Expression(makeType<int>(), 4));

	TypeChecker checker;

	if (checker.checkType<Expression>(exp, {}))
		FAIL();

	EXPECT_EQ(exp.getType(), makeType<int>());
}

TEST(TypeCheckTest, andOfBoolShouldProduceBool)	 // NOLINT
{
	Expression exp = Expression::op<OperationKind::add>(
			Type::unknown(),
			Expression(makeType<bool>(), true),
			Expression(makeType<bool>(), false));

	TypeChecker checker;

	if (checker.checkType<Expression>(exp, {}))
		FAIL();

	EXPECT_EQ(exp.getType(), makeType<bool>());
}

TEST(TypeCheckerTest, tupleExpressionType)	// NOLINT
{
	Expression exp(
			Type::unknown(),
			Tuple({ Expression(Type::Int(), ReferenceAccess("x")),
							Expression(Type::Float(), ReferenceAccess("y")) }));

	SymbolTable table;

	Member x("x", Type::Int(), TypePrefix::none());
	Member y("y", Type::Float(), TypePrefix::none());

	table.addSymbol(x);
	table.addSymbol(y);

	TypeChecker typeChecker;

	if (typeChecker.checkType<Tuple>(exp, table))
		FAIL();

	Type expected({ Type::Int(), Type::Float() });
	ASSERT_EQ(expected, exp.getType());
}
