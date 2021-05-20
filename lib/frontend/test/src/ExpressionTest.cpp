#include <gtest/gtest.h>
#include <modelica/frontend/AST.h>

using namespace modelica;
using namespace frontend;

TEST(AST, expressionConstantCanBeBuilt)	 // NOLINT
{
	SourceRange location = SourceRange::unknown();
	auto expression = Expression::constant(location, makeType<int>(), 3);

	EXPECT_TRUE(expression->isa<Constant>());
	EXPECT_TRUE(expression->get<Constant>()->isa<BuiltInType::Integer>());
	EXPECT_EQ(expression->get<Constant>()->get<BuiltInType::Integer>(), 3);
	EXPECT_EQ(expression->getType().get<BuiltInType>(), BuiltInType::Integer);
	EXPECT_EQ(expression->getType().size(), 1);
}

TEST(AST, expressionReferenceCanBeBuilt)	 // NOLINT
{
	SourceRange location = SourceRange::unknown();
	auto expression = Expression::reference(location, makeType<int>(), "x");

	EXPECT_TRUE(expression->isa<ReferenceAccess>());
	EXPECT_EQ(expression->get<ReferenceAccess>()->getName(), "x");
}

TEST(AST, expressionOperationCanBeBuilt)	 // NOLINT
{
	SourceRange location = SourceRange::unknown();
	auto constant = Expression::constant(location, makeType<int>(), 3);
	auto expression = Expression::operation(location, makeType<int>(), OperationKind::add,
																	llvm::ArrayRef({ constant->clone(), constant->clone() }));

	EXPECT_TRUE(expression->isa<Operation>());
	EXPECT_EQ(expression->get<Operation>()->getOperationKind(), OperationKind::add);
}

TEST(AST, expressionCallCanBeBuilt)	 // NOLINT
{
	SourceRange location = SourceRange::unknown();
	auto constant = Expression::constant(location, makeType<int>(), 3);

	auto expression = Expression::call(
			location, makeType<int>(),
	    Expression::reference(location, makeType<int>(), "Foo"),
			llvm::None);

	EXPECT_TRUE(expression->isa<Call>());
	EXPECT_EQ(expression->get<Call>()->getFunction()->get<ReferenceAccess>()->getName(), "Foo");
}

TEST(AST, expressionTupleCanBeBuilt)	 // NOLINT
{
	SourceRange location = SourceRange::unknown();
	auto constant = Expression::constant(location, makeType<int>(), 3);

	Type type(PackedType({ makeType<int>(), makeType<float>() }));
	auto expression = Expression::tuple(
			location, type,
			llvm::ArrayRef({
					Expression::reference(location, makeType<int>(), "x"),
					Expression::reference(location, makeType<float>(), "y") }));

	EXPECT_TRUE(expression->isa<Tuple>());
	EXPECT_EQ((*expression->get<Tuple>())[0]->get<ReferenceAccess>()->getName(), "x");
	EXPECT_EQ((*expression->get<Tuple>())[1]->get<ReferenceAccess>()->getName(), "y");
}
