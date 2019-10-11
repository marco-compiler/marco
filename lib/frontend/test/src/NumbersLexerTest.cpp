#include "gtest/gtest.h"

#include "modelica/NumbersLexer.hpp"

TEST(IntegerLexerTest, defaultShouldBeZero)
{
	modelica::IntegerLexer<10> lex;
	EXPECT_EQ(lex.get(), 0);
}

TEST(IntegerLexerTest, insertionShouldWork)
{
	modelica::IntegerLexer<10> lex;
	lex += 9;
	EXPECT_EQ(lex.get(), 9);
	lex += 7;
	EXPECT_EQ(lex.get(), 97);
}

TEST(FloatLexerTest, defaultShouldBeZero)
{
	modelica::FloatLexer<10> lex;
	EXPECT_NEAR(lex.get(), 0, 0.01);
	EXPECT_NEAR(lex.getUpperPart(), 0, 0.01);
}

TEST(FloatLexerTest, onlyUpperPartShouldWork)
{
	modelica::FloatLexer<10> lex;
	lex.addUpper(9);
	EXPECT_NEAR(lex.get(), 9, 0.01);
	lex.addUpper(7);
	EXPECT_NEAR(lex.get(), 97, 0.01);
}

TEST(FloatLexerTest, withoutExponentialShouldWork)
{
	modelica::FloatLexer<10> lex;
	lex.addUpper(9);
	lex.addLower(5);
	EXPECT_NEAR(lex.get(), 9.5, 0.01);
	lex.addLower(3);
	EXPECT_NEAR(lex.get(), 9.53, 0.01);
}

TEST(FloatLexerTest, positiveExponentialShouldWork)
{
	modelica::FloatLexer<10> lex;
	lex.addUpper(9);
	lex.addLower(5);
	lex.addExponential(0);
	EXPECT_NEAR(lex.get(), 9.5, 0.01);
	lex.addExponential(1);
	EXPECT_NEAR(lex.get(), 95, 0.01);
	lex.setSign(false);
	EXPECT_NEAR(lex.get(), 0.95, 0.01);
}
