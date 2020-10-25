#include "gtest/gtest.h"

#include "modelica/frontend/LexerStateMachine.hpp"
#include "modelica/utils/Lexer.hpp"

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

TEST(LexerStateMachineTest, singleCharIdentifierShouldParse)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("A");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::Ident);
	EXPECT_EQ(lexer.getLastIdentifier(), "A");
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

TEST(LexerStateMachineTest, integerFollowedByTokens)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("[1, 2; 3, 4]");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::LSquare);
	EXPECT_EQ(lexer.scan(), modelica::Token::Integer);
	EXPECT_EQ(lexer.scan(), modelica::Token::Comma);
	EXPECT_EQ(lexer.scan(), modelica::Token::Integer);
	EXPECT_EQ(lexer.scan(), modelica::Token::Semicolons);
	EXPECT_EQ(lexer.scan(), modelica::Token::Integer);
	EXPECT_EQ(lexer.scan(), modelica::Token::Comma);
	EXPECT_EQ(lexer.scan(), modelica::Token::Integer);
	EXPECT_EQ(lexer.scan(), modelica::Token::RSquare);
}

TEST(LexerStateMachineTest, unexpectedSymbolShouldFail)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("$");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::Error);
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, floatSubstractionShouldParse)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("4.0 - 5.0");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::FloatingPoint);
	EXPECT_EQ(lexer.scan(), modelica::Token::Minus);
	EXPECT_EQ(lexer.scan(), modelica::Token::FloatingPoint);
}

TEST(LexerStateMachineTest, minusShouldParse)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("-");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::Minus);
}

TEST(LexerStateMachineTest, multicharTokenShouldParse)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("== <= >= <> ./ .+ .- .* :=");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::OperatorEqual);
	EXPECT_EQ(lexer.scan(), modelica::Token::LessEqual);
	EXPECT_EQ(lexer.scan(), modelica::Token::GreaterEqual);
	EXPECT_EQ(lexer.scan(), modelica::Token::Different);
	EXPECT_EQ(lexer.scan(), modelica::Token::ElementWiseDivision);
	EXPECT_EQ(lexer.scan(), modelica::Token::ElementWiseSum);
	EXPECT_EQ(lexer.scan(), modelica::Token::ElementWiseMinus);
	EXPECT_EQ(lexer.scan(), modelica::Token::ElementWiseMultilpy);
	EXPECT_EQ(lexer.scan(), modelica::Token::Assignment);
	EXPECT_EQ(lexer.scan(), modelica::Token::End);
}

TEST(LexerStateMachineTest, singleDigitNumbers)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("7 8");
	Lex lexer(toParse);
	EXPECT_EQ(lexer.scan(), modelica::Token::Integer);
	EXPECT_EQ(lexer.getLastInt(), 7);
	EXPECT_EQ(lexer.scan(), modelica::Token::Integer);
	EXPECT_EQ(lexer.getLastInt(), 8);
}

TEST(LexerTest, Declaration)
{
	std::string s = "final parameter Real l(unit = \"m\") = "
									"0.0005714285714285715 \"Chip length in the x direction\";";
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;
	auto lexer = Lex(s);

	EXPECT_EQ(lexer.scan(), modelica::Token::FinalKeyword);
	EXPECT_EQ(lexer.scan(), modelica::Token::ParameterKeyword);
	EXPECT_EQ(lexer.scan(), modelica::Token::Ident);
	EXPECT_EQ(lexer.scan(), modelica::Token::Ident);
	EXPECT_EQ(lexer.scan(), modelica::Token::LPar);
	EXPECT_EQ(lexer.scan(), modelica::Token::Ident);
	EXPECT_EQ(lexer.scan(), modelica::Token::Equal);
	EXPECT_EQ(lexer.scan(), modelica::Token::String);
	EXPECT_EQ(lexer.scan(), modelica::Token::RPar);
	EXPECT_EQ(lexer.scan(), modelica::Token::Equal);
	EXPECT_EQ(lexer.scan(), modelica::Token::FloatingPoint);
	EXPECT_NEAR(lexer.getLastFloat(), 0.000571428557, 0.1);
	EXPECT_EQ(lexer.scan(), modelica::Token::String);
}

TEST(LexerTest, LexerOfSmallNumber)
{
	std::string s("0.000571428557");
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;
	auto lexer = Lex(s);
	EXPECT_EQ(lexer.scan(), modelica::Token::FloatingPoint);
	EXPECT_NEAR(lexer.getLastFloat(), 0.000571428557, 0.1);
}

TEST(LexerStateMachineTest, ifElseKeywords)
{
	using Lex = modelica::Lexer<modelica::ModelicaStateMachine>;

	std::string toParse("if true then 1 elseif false then 2 else 3");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), modelica::Token::IfKeyword);
	EXPECT_EQ(lexer.scan(), modelica::Token::TrueKeyword);
	EXPECT_EQ(lexer.scan(), modelica::Token::ThenKeyword);
	EXPECT_EQ(lexer.scan(), modelica::Token::Integer);
	EXPECT_EQ(lexer.scan(), modelica::Token::ElseIfKeyword);
	EXPECT_EQ(lexer.scan(), modelica::Token::FalseKeyword);
	EXPECT_EQ(lexer.scan(), modelica::Token::ThenKeyword);
	EXPECT_EQ(lexer.scan(), modelica::Token::Integer);
	EXPECT_EQ(lexer.scan(), modelica::Token::ElseKeyword);
	EXPECT_EQ(lexer.scan(), modelica::Token::Integer);
}
