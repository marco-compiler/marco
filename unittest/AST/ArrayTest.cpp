#include "gtest/gtest.h"
#include "marco/AST/AST.h"

using namespace ::marco;
using namespace ::marco::ast;

#define LOC SourceRange::unknown()

TEST(AST, array_1D)
{
  std::vector<std::unique_ptr<Expression>> values;

  values.push_back(Expression::constant(SourceRange::unknown(), makeType<BuiltInType::Integer>(), 0));
  values.push_back(Expression::constant(SourceRange::unknown(), makeType<BuiltInType::Integer>(), 0));
  values.push_back(Expression::constant(SourceRange::unknown(), makeType<BuiltInType::Integer>(), 0));

  auto expression = Expression::array(LOC, Type::unknown(), values);

  ASSERT_TRUE(expression->isa<Array>());
  EXPECT_EQ(expression->get<Array>()->size(), 3);
}
