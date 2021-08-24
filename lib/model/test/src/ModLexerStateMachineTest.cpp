#include "gtest/gtest.h"

#include "marco/model/ModLexerStateMachine.hpp"
#include "marco/utils/Lexer.hpp"

using namespace marco;

TEST(LexerStateMachineTest, checkDefaults)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

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
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("//asd");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::End);
	EXPECT_EQ(lexer.getCurrentLine(), 1);
	EXPECT_EQ(lexer.getCurrentColumn(), 6);
}

TEST(LexerStateMachineTest, multiLineCommentsShouldBeIgnored)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("/*asd\n\n*/");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::End);
	EXPECT_EQ(lexer.getCurrentLine(), 3);
	EXPECT_EQ(lexer.getCurrentColumn(), 3);
}

TEST(LexerStateMachineTest, integersShouldParse)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("1948");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::Integer);
	EXPECT_EQ(lexer.getLastInt(), 1948);
	EXPECT_EQ(lexer.scan(), ModToken::End);
}

TEST(LexerStateMachineTest, multipleIntegersShouldParse)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("1948 4000");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::Integer);
	EXPECT_EQ(lexer.getLastInt(), 1948);
	EXPECT_EQ(lexer.scan(), ModToken::Integer);
	EXPECT_EQ(lexer.getLastInt(), 4000);
	EXPECT_EQ(lexer.scan(), ModToken::End);
}

TEST(LexerStateMachineTest, floatShouldParse)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("19.48  17.3");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 19.48, 0.1);
	EXPECT_EQ(lexer.scan(), ModToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 17.3, 0.1);
	EXPECT_EQ(lexer.scan(), ModToken::End);
}

TEST(LexerStateMachineTest, exponentialShouldParse)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("2E4  3.0e-2");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 20000, 0.1);
	EXPECT_EQ(lexer.scan(), ModToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 0.03, 0.1);
	EXPECT_EQ(lexer.scan(), ModToken::End);
}

TEST(LexerStateMachineTest, dotOnlyFloatShouldParse)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("2.  3.0e-2");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 2, 0.1);
	EXPECT_EQ(lexer.scan(), ModToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 0.03, 0.1);
	EXPECT_EQ(lexer.scan(), ModToken::End);
}

TEST(LexerStateMachineTest, signOnlyFloatShouldFail)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("2E-");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::Error);
	EXPECT_EQ(lexer.scan(), ModToken::End);
}

TEST(LexerStateMachineTest, floatEMustBeFollowedBySignOrNumber)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("2E");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::Error);
	EXPECT_EQ(lexer.scan(), ModToken::End);
}

TEST(LexerStateMachineTest, identifierShouldParse)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("Asd\nDsa");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::Ident);
	EXPECT_EQ(lexer.getLastIdentifier(), "Asd");
	EXPECT_EQ(lexer.scan(), ModToken::Ident);
	EXPECT_EQ(lexer.getLastIdentifier(), "Dsa");
	EXPECT_EQ(lexer.scan(), ModToken::End);
}

TEST(LexerStateMachineTest, keywordsShouldParse)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("Asd\ninit");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::Ident);
	EXPECT_EQ(lexer.getLastIdentifier(), "Asd");
	EXPECT_EQ(lexer.scan(), ModToken::InitKeyword);
	EXPECT_EQ(lexer.scan(), ModToken::End);
}

TEST(LexerStateMachineTest, singleCharIdentifierShouldParse)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("A");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::Ident);
	EXPECT_EQ(lexer.getLastIdentifier(), "A");
	EXPECT_EQ(lexer.scan(), ModToken::End);
}

TEST(LexerStateMachineTest, symbolsShouldParse)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("*/ -");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::Multiply);
	EXPECT_EQ(lexer.scan(), ModToken::Division);
	EXPECT_EQ(lexer.scan(), ModToken::Minus);
	EXPECT_EQ(lexer.scan(), ModToken::End);
}

TEST(LexerStateMachineTest, integerFollowedByTokens)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("[1+ 2 3/ 4]");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::LSquare);
	EXPECT_EQ(lexer.scan(), ModToken::Integer);
	EXPECT_EQ(lexer.scan(), ModToken::Plus);
	EXPECT_EQ(lexer.scan(), ModToken::Integer);
	EXPECT_EQ(lexer.scan(), ModToken::Integer);
	EXPECT_EQ(lexer.scan(), ModToken::Division);
	EXPECT_EQ(lexer.scan(), ModToken::Integer);
	EXPECT_EQ(lexer.scan(), ModToken::RSquare);
}

TEST(LexerStateMachineTest, unexpectedSymbolShouldFail)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("$");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::Error);
	EXPECT_EQ(lexer.scan(), ModToken::End);
}

TEST(LexerStateMachineTest, multicharTokenShouldParse)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("== <= >=");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::OperatorEqual);
	EXPECT_EQ(lexer.scan(), ModToken::LessEqual);
	EXPECT_EQ(lexer.scan(), ModToken::GreaterEqual);
	EXPECT_EQ(lexer.scan(), ModToken::End);
}

TEST(LexerStateMachineTest, singleDigitNumbers)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("7 8");
	Lex lexer(toParse);
	EXPECT_EQ(lexer.scan(), ModToken::Integer);
	EXPECT_EQ(lexer.getLastInt(), 7);
	EXPECT_EQ(lexer.scan(), ModToken::Integer);
	EXPECT_EQ(lexer.getLastInt(), 8);
}

TEST(LexerStateMachineTest, backwardTest)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("backward");
	Lex lexer(toParse);
	EXPECT_EQ(lexer.scan(), ModToken::BackwardKeyword);
}

TEST(LexerStateMachineTest, constVectorTest)
{
	using Lex = marco::Lexer<ModLexerStateMachine>;

	std::string toParse("{1.4, 2.1, 3.9}");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), ModToken::LCurly);
	EXPECT_EQ(lexer.scan(), ModToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 1.4f, 0.1f);
	EXPECT_EQ(lexer.scan(), ModToken::Comma);
	EXPECT_EQ(lexer.scan(), ModToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 2.1f, 0.1f);
	EXPECT_EQ(lexer.scan(), ModToken::Comma);
	EXPECT_EQ(lexer.scan(), ModToken::Float);
	EXPECT_NEAR(lexer.getLastFloat(), 3.9f, 0.1f);
	EXPECT_EQ(lexer.scan(), ModToken::RCurly);
}
