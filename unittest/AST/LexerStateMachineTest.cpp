#include "gtest/gtest.h"

/*
#include "marco/AST/LexerStateMachine.h"
#include "marco/Utils/Lexer.h"

using namespace marco;
using namespace marco::ast;

TEST(LexerStateMachineTest, integerFollowedByTokens)
{
	using Lex = Lexer<ModelicaStateMachine>;

	std::string toParse("[1, 2; 3, 4]");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), Token::LSquare);
	EXPECT_EQ(lexer.scan(), Token::Integer);
	EXPECT_EQ(lexer.scan(), Token::Comma);
	EXPECT_EQ(lexer.scan(), Token::Integer);
	EXPECT_EQ(lexer.scan(), Token::Semicolons);
	EXPECT_EQ(lexer.scan(), Token::Integer);
	EXPECT_EQ(lexer.scan(), Token::Comma);
	EXPECT_EQ(lexer.scan(), Token::Integer);
	EXPECT_EQ(lexer.scan(), Token::RSquare);
}

TEST(LexerStateMachineTest, unexpectedSymbolShouldFail)
{
	using Lex = Lexer<ModelicaStateMachine>;

	std::string toParse("$");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), Token::Error);
	EXPECT_EQ(lexer.scan(), Token::End);
}

TEST(LexerStateMachineTest, floatSubstractionShouldParse)
{
	using Lex = Lexer<ModelicaStateMachine>;

	std::string toParse("4.0 - 5.0");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), Token::FloatingPoint);
	EXPECT_EQ(lexer.scan(), Token::Minus);
	EXPECT_EQ(lexer.scan(), Token::FloatingPoint);
}

TEST(LexerTest, Declaration)
{
	std::string s = "final parameter Real l(unit = \"m\") = "
									"0.0005714285714285715 \"Chip length in the x direction\";";
	using Lex = Lexer<ModelicaStateMachine>;
	auto lexer = Lex(s);

	EXPECT_EQ(lexer.scan(), Token::FinalKeyword);
	EXPECT_EQ(lexer.scan(), Token::ParameterKeyword);
	EXPECT_EQ(lexer.scan(), Token::Ident);
	EXPECT_EQ(lexer.scan(), Token::Ident);
	EXPECT_EQ(lexer.scan(), Token::LPar);
	EXPECT_EQ(lexer.scan(), Token::Ident);
	EXPECT_EQ(lexer.scan(), Token::Equal);
	EXPECT_EQ(lexer.scan(), Token::String);
	EXPECT_EQ(lexer.scan(), Token::RPar);
	EXPECT_EQ(lexer.scan(), Token::Equal);
	EXPECT_EQ(lexer.scan(), Token::FloatingPoint);
	EXPECT_NEAR(lexer.getLastFloat(), 0.000571428557, 0.1);
	EXPECT_EQ(lexer.scan(), Token::String);
}

TEST(LexerStateMachineTest, ifElseKeywords)
{
	using Lex = Lexer<ModelicaStateMachine>;

	std::string toParse("if true then 1 elseif false then 2 else 3");
	Lex lexer(toParse);

	EXPECT_EQ(lexer.scan(), Token::IfKeyword);
	EXPECT_EQ(lexer.scan(), Token::TrueKeyword);
	EXPECT_EQ(lexer.scan(), Token::ThenKeyword);
	EXPECT_EQ(lexer.scan(), Token::Integer);
	EXPECT_EQ(lexer.scan(), Token::ElseIfKeyword);
	EXPECT_EQ(lexer.scan(), Token::FalseKeyword);
	EXPECT_EQ(lexer.scan(), Token::ThenKeyword);
	EXPECT_EQ(lexer.scan(), Token::Integer);
	EXPECT_EQ(lexer.scan(), Token::ElseKeyword);
	EXPECT_EQ(lexer.scan(), Token::Integer);
}
*/
