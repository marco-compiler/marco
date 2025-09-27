#include "marco/AST/BaseModelica/AST.h"
#include "gtest/gtest.h"

using namespace ::marco;
using namespace ::marco::ast;

#define LOC SourceRange::unknown()

/*
TEST(AST, tuple_singleElement)
{
        auto expression = Expression::tuple(
      LOC, Type::unknown(),
      Expression::reference(LOC, makeType<BuiltInType::Integer>(), "x"));

  ASSERT_TRUE(expression->isa<Tuple>());
        EXPECT_EQ(expression->get<Tuple>()->size(), 1);
}

TEST(AST, tuple_multipleElements)
{
  std::vector<std::unique_ptr<Expression>> values;

        values.push_back(Expression::reference(LOC,
makeType<BuiltInType::Boolean>(), "x"));
        values.push_back(Expression::reference(LOC,
makeType<BuiltInType::Integer>(), "y"));
        values.push_back(Expression::reference(LOC,
makeType<BuiltInType::Real>(), "z"));

        auto expression = Expression::tuple(LOC, Type::unknown(), values);

  ASSERT_TRUE(expression->isa<Tuple>());
        EXPECT_EQ(expression->get<Tuple>()->size(), 3);
}
*/