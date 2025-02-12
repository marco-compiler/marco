#include "marco/Parser/FloatLexer.h"
#include "gtest/gtest.h"

using namespace ::marco;

TEST(FloatLexerTest, defaultShouldBeZero) {
  FloatLexer<10> lex;
  EXPECT_DOUBLE_EQ(lex.get(), 0);
  EXPECT_EQ(lex.getUpperPart(), 0);
}

TEST(FloatLexerTest, upperPartOnly) {
  FloatLexer<10> lex;

  lex.addUpper(9);
  EXPECT_DOUBLE_EQ(lex.get(), 9);

  lex.addUpper(7);
  EXPECT_DOUBLE_EQ(lex.get(), 97);
}

TEST(FloatLexerTest, noExponentSpecified) {
  FloatLexer<10> lex;

  lex.addUpper(9);
  lex.addLower(5);
  EXPECT_NEAR(lex.get(), 9.5, 0.1);

  lex.addLower(3);
  EXPECT_NEAR(lex.get(), 9.53, 0.01);
}

TEST(FloatLexerTest, exponent) {
  FloatLexer<10> lex;

  lex.addUpper(9);
  lex.addLower(5);
  lex.addExponential(0);
  EXPECT_NEAR(lex.get(), 9.5, 0.1);

  lex.addExponential(1);
  EXPECT_NEAR(lex.get(), 95, 1);

  lex.setSign(false);
  EXPECT_NEAR(lex.get(), 0.95, 0.01);
}

TEST(FloatLexerTest, smallPositiveNumber) {
  FloatLexer<10> lex;

  lex.addUpper(0);
  lex.addLower(0);
  lex.addLower(0);
  lex.addLower(0);
  lex.addLower(5);
  lex.addLower(7);
  lex.addLower(1);
  lex.addLower(4);
  lex.addLower(2);
  lex.addLower(8);
  EXPECT_NEAR(lex.get(), 0.000571428557, 0.000000001);
}
