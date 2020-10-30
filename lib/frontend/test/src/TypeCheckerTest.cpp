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

TEST(TypeCheckTest, sumOfIntShouldProduceInt)
{
	Expression exp = Expression::op<OperationKind::add>(
			Type::unknown(),
			Expression(makeType<int>(), 0),
			Expression(makeType<int>(), 4));

	TypeChecker checker;

	if (checker.checkType(exp, {}))
		FAIL();

	EXPECT_EQ(exp.getType(), makeType<int>());
}

TEST(TypeCheckTest, andOfBoolShouldProduceBool)
{
	Expression exp = Expression::op<OperationKind::add>(
			Type::unknown(),
			Expression(makeType<bool>(), true),
			Expression(makeType<bool>(), false));

	TypeChecker checker;

	if (checker.checkType(exp, {}))
		FAIL();

	EXPECT_EQ(exp.getType(), makeType<bool>());
}

TEST(StatementTypeCheck, directDerCall)
{
	// y := der(x)
	Expression expression = makeCall(
			Expression(Type::unknown(), ReferenceAccess("der")),
			Expression(Type::unknown(), ReferenceAccess("x")));

	Expression destination = Expression(Type::unknown(), ReferenceAccess("y"));
	Statement statement = Statement(destination, expression);

	TypeChecker checker;
	ASSERT_ERROR(checker.checkType(statement, {}), BadSemantic);
}

TEST(StatementTypeCheck, derInsideParameters)
{
	// z := x + foo(der(y))
	Expression x = Expression(Type::unknown(), ReferenceAccess("x"));
	Expression der = makeCall(
			Expression(Type::unknown(), ReferenceAccess("der")),
			Expression(Type::unknown(), ReferenceAccess("y")));
	Expression foo =
			makeCall(Expression(Type::unknown(), ReferenceAccess("foo")), der);

	Expression destination = Expression(Type::unknown(), ReferenceAccess("z"));

	Expression expression =
			Expression::op<OperationKind::add>(Type::unknown(), x, foo);
	Statement statement = Statement(destination, expression);

	TypeChecker checker;
	ASSERT_ERROR(checker.checkType(statement, {}), BadSemantic);
}
