#include "marco/AST/AST.h"
#include "gtest/gtest.h"

using namespace ::marco;
using namespace ::marco::ast;

#define LOC SourceRange::unknown()

/*
TEST(AST, array_1D)
{
  std::vector<std::unique_ptr<ASTNode>> values;

  values.push_back(std::make_unique<Constant>(LOC, 0));
  values.push_back(std::make_unique<Constant>(LOC, 0));
  values.push_back(std::make_unique<Constant>(LOC, 0));

  auto array = std::make_unique<Array>(LOC);
  array->setValues(values);

  ASSERT_TRUE(array->isa<Array>());
  EXPECT_EQ(array->get<Array>()->size(), 3);
}
*/