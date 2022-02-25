#include "gtest/gtest.h"
#include "marco/utils/NumbersLexer.h"

TEST(IntegerLexerTest, defaultShouldBeZero)
{
	marco::IntegerLexer<10> lex;
	EXPECT_EQ(lex.get(), 0);
}

TEST(IntegerLexerTest, insertionShouldWork)
{
	marco::IntegerLexer<10> lex;
	lex += 9;
	EXPECT_EQ(lex.get(), 9);
	lex += 7;
	EXPECT_EQ(lex.get(), 97);
}

TEST(FloatLexerTest, defaultShouldBeZero)
{
	marco::FloatLexer<10> lex;
	EXPECT_NEAR(lex.get(), 0, 0.01);
	EXPECT_NEAR(lex.getUpperPart(), 0, 0.01);
}

TEST(FloatLexerTest, onlyUpperPartShouldWork)
{
	marco::FloatLexer<10> lex;
	lex.addUpper(9);
	EXPECT_NEAR(lex.get(), 9, 0.01);
	lex.addUpper(7);
	EXPECT_NEAR(lex.get(), 97, 0.01);
}

TEST(FloatLexerTest, withoutExponentialShouldWork)
{
	marco::FloatLexer<10> lex;
	lex.addUpper(9);
	lex.addLower(5);
	EXPECT_NEAR(lex.get(), 9.5, 0.01);
	lex.addLower(3);
	EXPECT_NEAR(lex.get(), 9.53, 0.01);
}

TEST(FloatLexerTest, lessThanOneFloatsShoudlWork)
{
	marco::FloatLexer<10> lex;
	lex.addUpper(2);
	lex.addLower(0);
	lex.addLower(0);
	lex.addLower(2);
	EXPECT_NEAR(lex.get(), 2.002, 0.01);
	lex.addExponential(1);
	EXPECT_NEAR(lex.get(), 20.02, 0.01);
}

TEST(FloatLexerTest, positiveExponentialShouldWork)
{
	marco::FloatLexer<10> lex;
	lex.addUpper(9);
	lex.addLower(5);
	lex.addExponential(0);
	EXPECT_NEAR(lex.get(), 9.5, 0.01);
	lex.addExponential(1);
	EXPECT_NEAR(lex.get(), 95, 0.01);
	lex.setSign(false);
	EXPECT_NEAR(lex.get(), 0.95, 0.01);
}

TEST(FloatLexerTest, smallPositiveNumberShouldWokr)
{
	marco::FloatLexer<10> lex;
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
	EXPECT_NEAR(lex.get(), 0.000571428557, 1);
}