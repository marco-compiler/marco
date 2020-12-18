#include <gtest/gtest.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Error.h>
#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/Parser.hpp>
#include <modelica/frontend/Type.hpp>
#include <modelica/frontend/TypeChecker.hpp>
#include <modelica/utils/ErrorTest.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

TEST(TypeCheckTest, sumOfIntShouldProduceInt)	 // NOLINT
{
	SourcePosition location("-", 0, 0);
	Expression exp = Expression::op<OperationKind::add>(
			location,
			Type::unknown(),
			Expression(location, makeType<int>(), 0),
			Expression(location, makeType<int>(), 4));

	TypeChecker checker;

	if (checker.check<Expression>(exp))
		FAIL();

	EXPECT_EQ(exp.getType(), makeType<int>());
}

TEST(TypeCheckTest, andOfBoolShouldProduceBool)	 // NOLINT
{
	SourcePosition location("-", 0, 0);
	Expression exp = Expression::op<OperationKind::add>(
			location,
			Type::unknown(),
			Expression(location, makeType<bool>(), true),
			Expression(location, makeType<bool>(), false));

	TypeChecker checker;

	if (checker.check<Expression>(exp))
		FAIL();

	EXPECT_EQ(exp.getType(), makeType<bool>());
}

TEST(TypeCheckerTest, tupleExpressionType)	// NOLINT
{
	SourcePosition location("-", 0, 0);
	Expression exp(
			location,
			Type::unknown(),
			Tuple({ Expression(location, Type::Int(), ReferenceAccess("x")),
							Expression(location, Type::Float(), ReferenceAccess("y")) }));


	Member x("x", Type::Int(), TypePrefix::none());
	Member y("y", Type::Float(), TypePrefix::none());

	TypeChecker typeChecker;
	auto& symbolTable = typeChecker.getSymbolTable();
	TypeChecker::SymbolTableScope scope(symbolTable);

	symbolTable.insert(x.getName(), Symbol(x));
	symbolTable.insert(y.getName(), Symbol(y));

	if (typeChecker.check<Tuple>(exp))
		FAIL();

	Type expected({ Type::Int(), Type::Float() });
	ASSERT_EQ(expected, exp.getType());
}
