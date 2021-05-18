#include <gtest/gtest.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Error.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/Parser.hpp>
#include <modelica/frontend/passes/TypeCheckingPass.h>
#include <modelica/utils/ErrorTest.hpp>

using namespace llvm;
using namespace modelica;
using namespace frontend;
using namespace std;

TEST(TypeCheckTest, sumOfIntShouldProduceInt)	 // NOLINT
{
	auto exp = Expression::operation(
			SourcePosition::unknown(),
			Type::unknown(),
			OperationKind::add,
			llvm::ArrayRef({
					Expression::constant(SourcePosition::unknown(), makeType<int>(), 0),
					Expression::constant(SourcePosition::unknown(), makeType<int>(), 4)
			}));

	TypeChecker checker;

	if (checker.run<Expression>(*exp))
		FAIL();

	EXPECT_EQ(exp->getType(), makeType<int>());
}

TEST(TypeCheckTest, andOfBoolShouldProduceBool)	 // NOLINT
{
	auto exp = Expression::operation(
			SourcePosition::unknown(),
			Type::unknown(),
			OperationKind::add,
			llvm::ArrayRef({
					Expression::constant(SourcePosition::unknown(), makeType<bool>(), true),
					Expression::constant(SourcePosition::unknown(), makeType<bool>(), false)
			}));

	TypeChecker checker;

	if (checker.run<Expression>(*exp))
		FAIL();

	EXPECT_EQ(exp->getType(), makeType<bool>());
}

/*
TEST(TypeCheckerTest, tupleExpressionType)	// NOLINT
{
	SmallVector<Expression, 3> expressions;
	expressions.emplace_back(Expression::reference(SourcePosition::unknown(), makeType<int>(), "x"));
	expressions.emplace_back(Expression::reference(SourcePosition::unknown(), makeType<float>(), "y"));

	Expression exp = Expression::tuple(
			SourcePosition::unknown(),
			Type::unknown(),
			expressions);

	Member x(SourcePosition::unknown(), "x", makeType<int>(), TypePrefix::none());
	Member y(SourcePosition::unknown(), "y", makeType<float>(), TypePrefix::none());

	TypeChecker typeChecker;
	auto& symbolTable = typeChecker.getSymbolTable();
	TypeChecker::SymbolTableScope scope(symbolTable);

	symbolTable.insert(x.getName(), Symbol(x));
	symbolTable.insert(y.getName(), Symbol(y));

	if (typeChecker.check<Tuple>(exp))
		FAIL();

	Type expected({ makeType<int>(), makeType<float>() });
	ASSERT_EQ(expected, exp.getType());
}
*/