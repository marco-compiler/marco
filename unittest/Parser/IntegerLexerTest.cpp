#include "gtest/gtest.h"
#include "marco/Parser/IntegerLexer.h"

using namespace ::marco;

TEST(IntegerLexer, defaultShouldBeZero)
{
  IntegerLexer<10> lex;
  EXPECT_EQ(lex.get(), 0);
}

TEST(IntegerLexer, insertionShouldWork)
{
  IntegerLexer<10> lex;

  lex += 9;
  EXPECT_EQ(lex.get(), 9);

  lex += 7;
  EXPECT_EQ(lex.get(), 97);
}
