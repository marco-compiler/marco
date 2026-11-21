#include "marco/Lexer/Lexer.h"
#include "marco/Parser/Modelica/StateMachine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace ::marco;
using namespace ::marco::parser::modelica;

namespace {
/// Tokenize a string.
std::vector<Token> tokenize(llvm::StringRef source) {
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source);
  auto file = std::make_shared<SourceFile>("test.mo");
  file->setMemoryBuffer(buffer.get());
  lexer::Lexer<StateMachine> lexer(file);
  std::vector<Token> tokens;

  do {
    tokens.push_back(lexer.scan());
  } while (!tokens.back().isa<TokenKind::EndOfFile>() &&
           !tokens.back().isa<TokenKind::Error>());

  return tokens;
}

std::vector<TokenKind> kinds(llvm::StringRef source) {
  std::vector<TokenKind> result;

  for (const Token &token : tokenize(source)) {
    result.push_back(token.getKind());
  }

  return result;
}
} // namespace

TEST(ModelicaLexer, empty) {
  EXPECT_EQ(kinds(""), std::vector<TokenKind>{TokenKind::EndOfFile});
  EXPECT_EQ(kinds("  \n\t "), std::vector<TokenKind>{TokenKind::EndOfFile});
}

TEST(ModelicaLexer, byteOrderMark) {
  EXPECT_EQ(kinds("\xEF\xBB\xBFmodel"),
            (std::vector<TokenKind>{TokenKind::Model, TokenKind::EndOfFile}));
}

TEST(ModelicaLexer, keywordsAndIdentifiers) {
  auto tokens = tokenize("model Model _x1 time end");
  ASSERT_EQ(tokens.size(), 6u);

  EXPECT_EQ(tokens[0].getKind(), TokenKind::Model);
  EXPECT_EQ(tokens[1].getKind(), TokenKind::Identifier);
  EXPECT_EQ(tokens[1].getString(), "Model");
  EXPECT_EQ(tokens[2].getKind(), TokenKind::Identifier);
  EXPECT_EQ(tokens[2].getString(), "_x1");

  // 'time' is a keyword.
  EXPECT_EQ(tokens[3].getKind(), TokenKind::Time);
  EXPECT_EQ(tokens[4].getKind(), TokenKind::End);
}

TEST(ModelicaLexer, quotedIdentifiers) {
  auto tokens = tokenize(R"('x' '+foo' '13\'H' 'a\"b' 'a\?b')");
  ASSERT_EQ(tokens.size(), 6u);

  // The quotes are part of the identifier.
  EXPECT_EQ(tokens[0].getString(), "'x'");
  EXPECT_EQ(tokens[1].getString(), "'+foo'");
  EXPECT_EQ(tokens[2].getString(), R"('13\'H')");

  // Redundant escapes are the same as the non-escaped characters.
  EXPECT_EQ(tokens[3].getString(), R"('a"b')");
  EXPECT_EQ(tokens[4].getString(), "'a?b'");
}

TEST(ModelicaLexer, integers) {
  auto tokens = tokenize("0 42 9223372036854775807");
  ASSERT_EQ(tokens.size(), 4u);

  EXPECT_EQ(tokens[0].getKind(), TokenKind::Integer);
  EXPECT_EQ(tokens[0].getInt(), 0);
  EXPECT_EQ(tokens[1].getInt(), 42);
  EXPECT_EQ(tokens[2].getInt(), 9223372036854775807);
}

TEST(ModelicaLexer, integerTooLarge) {
  auto tokens = tokenize("9223372036854775808");
  ASSERT_EQ(tokens.back().getKind(), TokenKind::Error);
  EXPECT_EQ(tokens.back().getString(), "integer literal is too large");
}

TEST(ModelicaLexer, reals) {
  auto tokens = tokenize("1.5 2. .5 1e3 1.e-3 2.5E+2");
  ASSERT_EQ(tokens.size(), 7u);

  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(tokens[i].getKind(), TokenKind::Real);
  }

  EXPECT_DOUBLE_EQ(tokens[0].getFloat(), 1.5);
  EXPECT_DOUBLE_EQ(tokens[1].getFloat(), 2);
  EXPECT_DOUBLE_EQ(tokens[2].getFloat(), 0.5);
  EXPECT_DOUBLE_EQ(tokens[3].getFloat(), 1000);
  EXPECT_DOUBLE_EQ(tokens[4].getFloat(), 0.001);
  EXPECT_DOUBLE_EQ(tokens[5].getFloat(), 250);
}

TEST(ModelicaLexer, missingExponentDigits) {
  auto tokens = tokenize("1e+x");
  ASSERT_EQ(tokens.back().getKind(), TokenKind::Error);
  EXPECT_EQ(tokens.back().getString(), "missing digits in the exponent");
}

TEST(ModelicaLexer, realWithTrailingDot) {
  // '2.' is a lexical unit, thus '2./x' is '2.0 / x'.
  EXPECT_EQ(kinds("2./x"),
            (std::vector<TokenKind>{TokenKind::Real, TokenKind::Slash,
                                    TokenKind::Identifier,
                                    TokenKind::EndOfFile}));

  EXPECT_EQ(kinds("2 ./x"),
            (std::vector<TokenKind>{TokenKind::Integer, TokenKind::SlashEW,
                                    TokenKind::Identifier,
                                    TokenKind::EndOfFile}));
}

TEST(ModelicaLexer, strings) {
  auto tokens = tokenize(R"("abc" "a\"b\n" "")");
  ASSERT_EQ(tokens.size(), 4u);

  EXPECT_EQ(tokens[0].getKind(), TokenKind::String);
  EXPECT_EQ(tokens[0].getString(), "abc");
  EXPECT_EQ(tokens[1].getString(), "a\"b\n");
  EXPECT_EQ(tokens[2].getString(), "");
}

TEST(ModelicaLexer, unterminatedString) {
  auto tokens = tokenize("\"abc");
  ASSERT_EQ(tokens.back().getKind(), TokenKind::Error);
  EXPECT_EQ(tokens.back().getString(), "unterminated string");
}

TEST(ModelicaLexer, invalidEscape) {
  auto tokens = tokenize(R"("a\qb")");
  ASSERT_EQ(tokens.back().getKind(), TokenKind::Error);
  EXPECT_EQ(tokens.back().getString(), "invalid escape sequence in string");
}

TEST(ModelicaLexer, symbols) {
  EXPECT_EQ(kinds("+ .+ - .- * .* / ./ ^ .^ = == <> < <= > >= := , ; : . "
                  "( ) [ ] { }"),
            (std::vector<TokenKind>{
                TokenKind::Plus,      TokenKind::PlusEW,
                TokenKind::Minus,     TokenKind::MinusEW,
                TokenKind::Star,      TokenKind::StarEW,
                TokenKind::Slash,     TokenKind::SlashEW,
                TokenKind::Caret,     TokenKind::CaretEW,
                TokenKind::Equal,     TokenKind::EqualEqual,
                TokenKind::NotEqual,  TokenKind::Less,
                TokenKind::LessEqual, TokenKind::Greater,
                TokenKind::GreaterEqual, TokenKind::Assignment,
                TokenKind::Comma,     TokenKind::Semicolon,
                TokenKind::Colon,     TokenKind::Dot,
                TokenKind::LParen,    TokenKind::RParen,
                TokenKind::LSquare,   TokenKind::RSquare,
                TokenKind::LCurly,    TokenKind::RCurly,
                TokenKind::EndOfFile}));
}

TEST(ModelicaLexer, comments) {
  EXPECT_EQ(kinds("a // comment\n /* multi\nline */ b"),
            (std::vector<TokenKind>{TokenKind::Identifier,
                                    TokenKind::Identifier,
                                    TokenKind::EndOfFile}));
}

TEST(ModelicaLexer, unterminatedComment) {
  auto tokens = tokenize("a /* comment");
  ASSERT_EQ(tokens.back().getKind(), TokenKind::Error);
  EXPECT_EQ(tokens.back().getString(), "unterminated comment");
}

TEST(ModelicaLexer, unexpectedCharacter) {
  auto tokens = tokenize("a # b");
  ASSERT_EQ(tokens.back().getKind(), TokenKind::Error);
  EXPECT_EQ(tokens.back().getString(), "unexpected character '#'");
}

TEST(ModelicaLexer, locations) {
  auto tokens = tokenize("model M\n  Real xy;\nend M;");

  // 'xy'
  const Token &token = tokens[3];
  ASSERT_EQ(token.getKind(), TokenKind::Identifier);
  EXPECT_EQ(token.getLocation().begin.line, 2);
  EXPECT_EQ(token.getLocation().begin.column, 8);
  EXPECT_EQ(token.getLocation().end.line, 2);
  EXPECT_EQ(token.getLocation().end.column, 9);
}
