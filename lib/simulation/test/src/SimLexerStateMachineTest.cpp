#include "gtest/gtest.h"

#include "modelica/simulation/SimLexerStateMachine.hpp"
#include "modelica/utils/Lexer.hpp"

using namespace modelica;

TEST(LexerStateMachineTest, checkDefaults)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("bb");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.getLastInt(), 0);
	EXPECT_EQ(lexer.getLastFloat(), 0.0);
	EXPECT_EQ(lexer.getLastIdentifier(), "");
	EXPECT_EQ(lexer.getCurrentLine(), 1);
	EXPECT_EQ(lexer.getCurrentColumn(), 0);
}

TEST(LexerStateMachineTest, singleLineCommentsShouldBeIgnored)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("//asd");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::End);
	EXPECT_EQ(lexer.getCurrentLine(), 1);
	EXPECT_EQ(lexer.getCurrentColumn(), 6);
}

TEST(LexerStateMachineTest, multiLineCommentsShouldBeIgnored)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("/*asd\n\n*/");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::End);
	EXPECT_EQ(lexer.getCurrentLine(), 3);
	EXPECT_EQ(lexer.getCurrentColumn(), 3);
}

TEST(LexerStateMachineTest, integersShouldParse)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("1948");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::Integer);
	EXPECT_EQ(lexer.getLastInt(), 1948);
	EXPECT_EQ(lexer.scan(), SimToken::End);
}

TEST(LexerStateMachineTest, multipleIntegersShouldParse)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("1948 4000");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::Integer);
	EXPECT_EQ(lexer.getLastInt(), 1948);
	EXPECT_EQ(lexer.scan(), SimToken::Integer);
	EXPECT_EQ(lexer.getLastInt(), 4000);
	EXPECT_EQ(lexer.scan(), SimToken::End);
}

TEST(LexerStateMachineTest, floatShouldParse)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("19.48  17.3");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 19.48, 0.1);
	EXPECT_EQ(lexer.scan(), SimToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 17.3, 0.1);
	EXPECT_EQ(lexer.scan(), SimToken::End);
}

TEST(LexerStateMachineTest, exponentialShouldParse)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("2E4  3.0e-2");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 20000, 0.1);
	EXPECT_EQ(lexer.scan(), SimToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 0.03, 0.1);
	EXPECT_EQ(lexer.scan(), SimToken::End);
}

TEST(LexerStateMachineTest, dotOnlyFloatShouldParse)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("2.  3.0e-2");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 2, 0.1);
	EXPECT_EQ(lexer.scan(), SimToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 0.03, 0.1);
	EXPECT_EQ(lexer.scan(), SimToken::End);
}

TEST(LexerStateMachineTest, signOnlyFloatShouldFail)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("2E-");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::Error);
	EXPECT_EQ(lexer.scan(), SimToken::End);
}

TEST(LexerStateMachineTest, floatEMustBeFollowedBySignOrNumber)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("2E");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::Error);
	EXPECT_EQ(lexer.scan(), SimToken::End);
}

TEST(LexerStateMachineTest, identifierShouldParse)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("Asd\nDsa");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::Ident);
	EXPECT_EQ(lexer.getLastIdentifier(), "Asd");
	EXPECT_EQ(lexer.scan(), SimToken::Ident);
	EXPECT_EQ(lexer.getLastIdentifier(), "Dsa");
	EXPECT_EQ(lexer.scan(), SimToken::End);
}

TEST(LexerStateMachineTest, keywordsShouldParse)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("Asd\ninit");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::Ident);
	EXPECT_EQ(lexer.getLastIdentifier(), "Asd");
	EXPECT_EQ(lexer.scan(), SimToken::InitKeyword);
	EXPECT_EQ(lexer.scan(), SimToken::End);
}

TEST(LexerStateMachineTest, singleCharIdentifierShouldParse)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("A");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::Ident);
	EXPECT_EQ(lexer.getLastIdentifier(), "A");
	EXPECT_EQ(lexer.scan(), SimToken::End);
}

TEST(LexerStateMachineTest, symbolsShouldParse)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("*/ -");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::Multiply);
	EXPECT_EQ(lexer.scan(), SimToken::Division);
	EXPECT_EQ(lexer.scan(), SimToken::Minus);
	EXPECT_EQ(lexer.scan(), SimToken::End);
}

TEST(LexerStateMachineTest, integerFollowedByTokens)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("[1+ 2 3/ 4]");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::LSquare);
	EXPECT_EQ(lexer.scan(), SimToken::Integer);
	EXPECT_EQ(lexer.scan(), SimToken::Plus);
	EXPECT_EQ(lexer.scan(), SimToken::Integer);
	EXPECT_EQ(lexer.scan(), SimToken::Integer);
	EXPECT_EQ(lexer.scan(), SimToken::Division);
	EXPECT_EQ(lexer.scan(), SimToken::Integer);
	EXPECT_EQ(lexer.scan(), SimToken::RSquare);
}

TEST(LexerStateMachineTest, unexpectedSymbolShouldFail)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("$");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::Error);
	EXPECT_EQ(lexer.scan(), SimToken::End);
}

TEST(LexerStateMachineTest, multicharTokenShouldParse)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("== <= >=");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::OperatorEqual);
	EXPECT_EQ(lexer.scan(), SimToken::LessEqual);
	EXPECT_EQ(lexer.scan(), SimToken::GreaterEqual);
	EXPECT_EQ(lexer.scan(), SimToken::End);
}

TEST(LexerStateMachineTest, singleDigitNumbers)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("7 8");
	Lex lexer(toParse);
	EXPECT_EQ(lexer.scan(), SimToken::Integer);
	EXPECT_EQ(lexer.getLastInt(), 7);
	EXPECT_EQ(lexer.scan(), SimToken::Integer);
	EXPECT_EQ(lexer.getLastInt(), 8);
}

TEST(LexerStateMachineTest, constVectorTest)
{
	using Lex = modelica::Lexer<SimLexerStateMachine>;

	std::string toParse("{1.4, 2.1, 3.9}");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), SimToken::LCurly);
	EXPECT_EQ(lexer.scan(), SimToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 1.4f, 0.1f);
	EXPECT_EQ(lexer.scan(), SimToken::Comma);
	EXPECT_EQ(lexer.scan(), SimToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 2.1f, 0.1f);
	EXPECT_EQ(lexer.scan(), SimToken::Comma);
	EXPECT_EQ(lexer.scan(), SimToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 3.9f, 0.1f);
	EXPECT_EQ(lexer.scan(), SimToken::RCurly);
}
