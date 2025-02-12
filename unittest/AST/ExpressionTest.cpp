#include "marco/AST/AST.h"
#include "gtest/gtest.h"

using namespace ::marco;
using namespace ::marco::ast;

#define LOC SourceRange::unknown()

/*
TEST(AST, expression_array)
{
  std::vector<std::unique_ptr<ASTNode>> values;

  auto value0 = std::make_unique<Constant>(LOC);
  value0->setValue(false);
  values.push_back(std::move(value0));

  auto value1 = std::make_unique<Constant>(LOC);
  value1->setValue(1);
  values.push_back(std::move(value1));

  auto value2 = std::make_unique<Constant>(LOC);
  value2->setValue(2);
  values.push_back(std::move(value2));

  auto node = std::make_unique<Array>(LOC);
  node->setValues(values);

  ASSERT_TRUE(node->isa<Array>());
  EXPECT_EQ(node->cast<Array>()->size(), 3);
}

TEST(AST, expression_call)
{
  auto node = std::make_unique<Call>(LOC);

  auto callee = std::make_unique<ReferenceAccess>(LOC);
  callee->setName("Foo");
  node->setCallee(std::move(callee));

  ASSERT_TRUE(node->isa<Call>());
  EXPECT_EQ(node->cast<Call>()->getCallee()->cast<ReferenceAccess>()->getName(),
"Foo");
}

TEST(AST, expression_constant)
{
        auto node = std::make_unique<Constant>(LOC);
  node->setValue(0);

        ASSERT_TRUE(node->isa<Constant>());
        EXPECT_EQ(node->cast<Constant>()->as<int64_t>(), 0);
}

TEST(AST, expression_reference)
{
        auto node = std::make_unique<ReferenceAccess>(LOC);
  node->setName("x");

        ASSERT_TRUE(node->isa<ReferenceAccess>());
        EXPECT_EQ(node->cast<ReferenceAccess>()->getName(), "x");
}

TEST(AST, expression_operation)
{
  std::vector<std::unique_ptr<ASTNode>> args;

  auto arg0 = std::make_unique<ReferenceAccess>(LOC);
  arg0->setName("x");
  args.push_back(std::move(arg0));

  auto arg1 = std::make_unique<ReferenceAccess>(LOC);
  arg1->setName("y");
  args.push_back(std::move(arg1));

        auto node = std::make_unique<Operation>(LOC);
  node->setOperationKind(OperationKind::add);
  node->setArguments(args);

        ASSERT_TRUE(node->isa<Operation>());
        EXPECT_EQ(node->cast<Operation>()->getOperationKind(),
OperationKind::add);
}

TEST(AST, expression_tuple)
{
  std::vector<std::unique_ptr<ASTNode>> values;

  auto value0 = std::make_unique<ReferenceAccess>(LOC);
  value0->setName("x");
  values.push_back(std::move(value0));

  auto value1 = std::make_unique<ReferenceAccess>(LOC);
  value1->setName("y");
  values.push_back(std::move(value1));

  auto value2 = std::make_unique<ReferenceAccess>(LOC);
  value2->setName("z");
  values.push_back(std::move(value2));

  auto node = std::make_unique<Tuple>(LOC);
  node->setExpressions(values);

  ASSERT_TRUE(node->isa<Tuple>());
  EXPECT_EQ((*node->cast<Tuple>()).getExpression(0)->cast<ReferenceAccess>()->getName(),
"x");
  EXPECT_EQ((*node->cast<Tuple>()).getExpression(1)->cast<ReferenceAccess>()->getName(),
"y");
  EXPECT_EQ((*node->cast<Tuple>()).getExpression(2)->cast<ReferenceAccess>()->getName(),
"z");
}
*/