#include "gtest/gtest.h"

#include "modelica/Lexer.hpp"
#include "modelica/LexerStateMachine.hpp"

TEST(LexerStateMachineTest, checkDefaults)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("bb");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.getLastInt(), 0);
	EXPECT_EQ(lexer.getLastFloat(), 0.0);
	EXPECT_EQ(lexer.getLastString(), "");
	EXPECT_EQ(lexer.getLastIdentifier(), "");
	EXPECT_EQ(lexer.getCurrentLine(), 1);
	EXPECT_EQ(lexer.getCurrentColumn(), 0);
}

TEST(LexerStateMachineTest, singleLineCommentsShouldBeIgnored)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("//asd");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::End);
	EXPECT_EQ(lexer.getCurrentLine(), 1);
	EXPECT_EQ(lexer.getCurrentColumn(), 6);
}

TEST(LexerStateMachineTest, multiLineCommentsShouldBeIgnored)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("/*asd\n\n*/");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::End);
	EXPECT_EQ(lexer.getCurrentLine(), 3);
	EXPECT_EQ(lexer.getCurrentColumn(), 3);
}

TEST(LexerStateMachineTest, integersShouldParse)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("1948");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::Integer);
	EXPECT_EQ(lexer.getLastInt(), 1948);
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, multipleIntegersShouldParse)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("1948 4000");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::Integer);
	EXPECT_EQ(lexer.getLastInt(), 1948);
	EXPECT_EQ(lexer.scan(), modelica::Token::Integer);
	EXPECT_EQ(lexer.getLastInt(), 4000);
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, floatShouldParse)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("19.48  17.3");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::FloatingPoint);
	EXPECT_NEAR(lexer.getLastFloat(), 19.48, 0.1);
	EXPECT_EQ(lexer.scan(), modelica::Token::FloatingPoint);
	EXPECT_NEAR(lexer.getLastFloat(), 17.3, 0.1);
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, exponentialShouldParse)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("2E4  3.0e-2");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::FloatingPoint);
	EXPECT_NEAR(lexer.getLastFloat(), 20000, 0.1);
	EXPECT_EQ(lexer.scan(), modelica::Token::FloatingPoint);
	EXPECT_NEAR(lexer.getLastFloat(), 0.03, 0.1);
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, dotOnlyFloatShouldParse)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("2.  3.0e-2");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::FloatingPoint);
	EXPECT_NEAR(lexer.getLastFloat(), 2, 0.1);
	EXPECT_EQ(lexer.scan(), modelica::Token::FloatingPoint);
	EXPECT_NEAR(lexer.getLastFloat(), 0.03, 0.1);
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, signOnlyFloatShouldFail)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("2E-");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::Error);
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, floatEMustBeFollowedBySignOrNumber)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("2E");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::Error);
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, stringsShouldParse)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("\"asd\"  \"another\"");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::String);
	EXPECT_EQ(lexer.getLastString(), "asd");
	EXPECT_EQ(lexer.scan(), modelica::Token::String);
	EXPECT_EQ(lexer.getLastString(), "another");
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, specialCaractersShouldWork)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("\"\\\"\\n\\r\\t\\v\\?\"");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::String);
	EXPECT_EQ(lexer.getLastString(), "\"\n\r\t\v?");
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, malformedStringsShouldReturnError)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("\"");
	Lex lexer(toParse);
	EXPECT_EQ(lexer.scan(), modelica::Token::Error);
}

TEST(LexerStateMachineTest, identifierShouldParse)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("Asd\nDsa");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::Ident);
	EXPECT_EQ(lexer.getLastIdentifier(), "Asd");
	EXPECT_EQ(lexer.scan(), modelica::Token::Ident);
	EXPECT_EQ(lexer.getLastIdentifier(), "Dsa");
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, keywordsShouldParse)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("Asd\nfinal");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::Ident);
	EXPECT_EQ(lexer.getLastIdentifier(), "Asd");
	EXPECT_EQ(lexer.scan(), modelica::Token::FinalKeyword);
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, qIdentShouldParse)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("'Asd'");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::Ident);
	EXPECT_EQ(lexer.getLastIdentifier(), "Asd");
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, qIdentShouldParseWithEscapedChars)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("'Asd\\'\\n'");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::Ident);
	EXPECT_EQ(lexer.getLastIdentifier(), "Asd'\n");
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, symbolsShouldParse)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("*/ -");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::Multiply);
	EXPECT_EQ(lexer.scan(), modelica::Token::Division);
	EXPECT_EQ(lexer.scan(), modelica::Token::Minus);
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, unexpectedSymbolShouldFail)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("^");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::Error);
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}
