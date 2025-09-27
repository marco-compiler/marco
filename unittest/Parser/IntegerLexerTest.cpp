#include "marco/Lexer/IntegerLexer.h"
#include "gtest/gtest.h"

using namespace ::marco::lexer;

TEST(IntegerLexer, defaultShouldBeZero) {
  IntegerLexer<10> lex;
  EXPECT_EQ(lex.get(), 0);
}

TEST(IntegerLexer, insertionShouldWork) {
  IntegerLexer<10> lex;

  lex += 9;
  EXPECT_EQ(lex.get(), 9);

  lex += 7;
  EXPECT_EQ(lex.get(), 97);
}
