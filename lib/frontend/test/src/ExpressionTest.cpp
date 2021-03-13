#include <gtest/gtest.h>

#include <modelica/frontend/Constant.hpp>
#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/Type.hpp>

using namespace modelica;

TEST(AST, expressionConstantCanBeBuilt)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();
	Expression exp = Expression::constant(location, makeType<int>(), 3);

	EXPECT_TRUE(exp.isA<Constant>());
	EXPECT_TRUE(exp.get<Constant>().isA<BuiltInType::Integer>());
	EXPECT_EQ(exp.get<Constant>().get<BuiltInType::Integer>(), 3);
	EXPECT_EQ(exp.getType().get<BuiltInType>(), BuiltInType::Integer);
	EXPECT_EQ(exp.getType().size(), 1);
}

TEST(AST, expressionReferenceCanBeBuilt)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();
	Expression exp = Expression::reference(location, makeType<int>(), "x");

	EXPECT_TRUE(exp.isA<ReferenceAccess>());
	EXPECT_EQ(exp.get<ReferenceAccess>().getName(), "x");
}

TEST(AST, expressionOperationCanBeBuilt)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();
	Expression constant = Expression::constant(location, makeType<int>(), 3);
	Expression exp = Expression::operation(location, makeType<int>(), OperationKind::add, constant, constant);

	EXPECT_TRUE(exp.isA<Operation>());
	EXPECT_EQ(exp.get<Operation>().getKind(), OperationKind::add);
}

TEST(AST, expressionCallCanBeBuilt)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();
	Expression constant = Expression::constant(location, makeType<int>(), 3);
	Expression exp = Expression::call(location, makeType<int>(), Expression::reference(location, makeType<int>(), "Foo"));

	EXPECT_TRUE(exp.isA<Call>());
	EXPECT_EQ(exp.get<Call>().getFunction().get<ReferenceAccess>().getName(), "Foo");
}

TEST(AST, expressionTupleCanBeBuilt)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();
	Expression constant = Expression::constant(location, makeType<int>(), 3);

	Type type({ makeType<int>(), makeType<float>() });
	Expression exp = Expression::tuple(location, type,
																		 Expression::reference(location, makeType<int>(), "x"),
																		 Expression::reference(location, makeType<float>(), "y"));

	EXPECT_TRUE(exp.isA<Tuple>());
	EXPECT_EQ(exp.get<Tuple>()[0].get<ReferenceAccess>().getName(), "x");
	EXPECT_EQ(exp.get<Tuple>()[1].get<ReferenceAccess>().getName(), "y");
}
