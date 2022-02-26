#include "gtest/gtest.h"
#include "marco/Utils/Lexer.h"
#include <algorithm>

enum class TestToken
{
	Begin,
	Ok,
	None,
	End
};

class TestStateMachine
{
	public:
	using Token = TestToken;
	TestStateMachine(char c): current(c) {}
	Token getCurrent() const { return currentTok; }
	Token step(char c)
	{
		current = c;
		if (c != '\0')
			currentTok = Token::Ok;
		else
			currentTok = Token::End;

		return currentTok;
	}

	private:
	char current;
	Token currentTok{ Token::Begin };
};

TEST(LexerTest, TestDefaultValues)
{
	std::string s("");
	auto lexer = marco::Lexer<TestStateMachine>(s.begin());

	EXPECT_EQ(lexer.getCurrent(), TestToken::Begin);
}

TEST(LexerTest, NullTerminatorShouldReturnEnd)
{
	std::string s("");
	auto lexer = marco::Lexer<TestStateMachine>(s.begin());

	EXPECT_EQ(lexer.scan(), TestToken::End);
}

TEST(LexerTest, CharPointerShouldCompile)
{
	std::string s("");
	auto lexer = marco::Lexer<TestStateMachine>(&(s[0]));

	EXPECT_EQ(lexer.scan(), TestToken::End);
}

TEST(LexerTest, LexerIsAdvanceble)
{
	std::string s("bb");
	using Lex = marco::Lexer<TestStateMachine>;
	auto lexer = Lex(s.begin());
	EXPECT_EQ(lexer.scan(), TestToken::Ok);
	EXPECT_EQ(lexer.scan(), TestToken::End);
	EXPECT_EQ(lexer.scan(), TestToken::End);
}

TEST(LexerTest, LexerShouldAcceptStrings)
{
	std::string s("bb");
	using Lex = marco::Lexer<TestStateMachine>;
	auto lexer = Lex(s);
	EXPECT_EQ(lexer.scan(), TestToken::Ok);
	EXPECT_EQ(lexer.scan(), TestToken::End);
	EXPECT_EQ(lexer.scan(), TestToken::End);
}

TEST(LexerTest, LexerOFStringEndShouldReturnEnd)
{
	std::string s("bb");
	using Lex = marco::Lexer<TestStateMachine>;
	auto lexer = Lex(s.end());
	EXPECT_EQ(lexer.scan(), TestToken::End);
	EXPECT_EQ(lexer.scan(), TestToken::End);
}

TEST(LexerTest, LexerIsAIterator)
{
	std::string s("bb");
	std::vector<TestToken> expected;
	std::vector<bool> output;
	output.resize(s.length());
	expected.push_back(TestToken::Begin);
	expected.push_back(TestToken::Ok);
	using Lex = marco::Lexer<TestStateMachine>;
	auto lexer = Lex(s.begin());
	int checks = 0;

	EXPECT_EQ(TestToken::Begin, *lexer.begin());

	std::transform(
			lexer.begin(),
			lexer.end(),
			expected.begin(),
			output.begin(),
			[&checks](const auto& first, const auto& second) {
				checks++;
				EXPECT_EQ(first, second);
				return !(first == second);
			});
	EXPECT_EQ(checks, 2);
	EXPECT_EQ(lexer.getCurrent(), TestToken::End);
}
