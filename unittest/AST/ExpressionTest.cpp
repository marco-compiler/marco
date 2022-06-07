#include "gtest/gtest.h"
#include "marco/AST/AST.h"

using namespace ::marco;
using namespace ::marco::ast;

#define LOC SourceRange::unknown()

TEST(AST, expression_array)
{
  std::vector<std::unique_ptr<Expression>> values;

  values.push_back(Expression::constant(LOC, makeType<BuiltInType::Boolean>(), false));
  values.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 1));
  values.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 2));

  auto expression = Expression::array(LOC, makeType<BuiltInType::Real>(3), values);

  ASSERT_TRUE(expression->isa<Array>());
  EXPECT_EQ(expression->get<Array>()->size(), 3);
}

TEST(AST, expression_call)
{
  auto constant = Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0);

  auto expression = Expression::call(
      LOC, makeType<BuiltInType::Integer>(),
      Expression::reference(LOC, makeType<BuiltInType::Integer>(), "Foo"),
      llvm::None);

  ASSERT_TRUE(expression->isa<Call>());
  EXPECT_EQ(expression->get<Call>()->getFunction()->get<ReferenceAccess>()->getName(), "Foo");
}

TEST(AST, expression_constant)
{
	auto expression = Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0);

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>());

	ASSERT_TRUE(expression->isa<Constant>());
	ASSERT_TRUE(expression->get<Constant>()->isa<BuiltInType::Integer>());
	EXPECT_EQ(expression->get<Constant>()->get<BuiltInType::Integer>(), 0);
}

TEST(AST, expression_reference)
{
	auto expression = Expression::reference(LOC, makeType<BuiltInType::Integer>(), "x");

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>());

	ASSERT_TRUE(expression->isa<ReferenceAccess>());
	EXPECT_EQ(expression->get<ReferenceAccess>()->getName(), "x");
}

TEST(AST, expression_operation)
{
  std::vector<std::unique_ptr<Expression>> args;

	args.push_back(Expression::reference(LOC, makeType<BuiltInType::Integer>(), "x"));
  args.push_back(Expression::reference(LOC, makeType<BuiltInType::Integer>(), "y"));

	auto expression = Expression::operation(LOC, makeType<BuiltInType::Integer>(), OperationKind::add, args);

	ASSERT_TRUE(expression->isa<Operation>());
	EXPECT_EQ(expression->get<Operation>()->getOperationKind(), OperationKind::add);
}

TEST(AST, expression_tuple)
{
  std::vector<std::unique_ptr<Expression>> values;

  values.push_back(Expression::reference(LOC, makeType<BuiltInType::Boolean>(), "x"));
  values.push_back(Expression::reference(LOC, makeType<BuiltInType::Integer>(), "y"));
  values.push_back(Expression::reference(LOC, makeType<BuiltInType::Real>(), "z"));

  Type type(PackedType({
      makeType<BuiltInType::Boolean>(),
      makeType<BuiltInType::Integer>(),
      makeType<BuiltInType::Real>()
  }));

  auto expression = Expression::tuple(LOC, type, values);

  ASSERT_TRUE(expression->isa<Tuple>());
  EXPECT_EQ((*expression->get<Tuple>())[0]->get<ReferenceAccess>()->getName(), "x");
  EXPECT_EQ((*expression->get<Tuple>())[1]->get<ReferenceAccess>()->getName(), "y");
  EXPECT_EQ((*expression->get<Tuple>())[2]->get<ReferenceAccess>()->getName(), "z");
}
