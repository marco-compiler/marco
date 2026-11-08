#include "marco/Parser/Modelica/Lexer.h"
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
  Lexer lexer(file);
  return lexer.tokenize();
}

std::vector<TokenKind> kinds(llvm::StringRef source) {
  std::vector<TokenKind> result;

  for (const Token &token : tokenize(source)) {
    result.push_back(token.kind);
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

  EXPECT_EQ(tokens[0].kind, TokenKind::Model);
  EXPECT_EQ(tokens[1].kind, TokenKind::Identifier);
  EXPECT_EQ(tokens[1].text, "Model");
  EXPECT_EQ(tokens[2].kind, TokenKind::Identifier);
  EXPECT_EQ(tokens[2].text, "_x1");

  // 'time' is a keyword.
  EXPECT_EQ(tokens[3].kind, TokenKind::Time);
  EXPECT_EQ(tokens[4].kind, TokenKind::End);
}

TEST(ModelicaLexer, quotedIdentifiers) {
  auto tokens = tokenize(R"('x' '+foo' '13\'H' 'a\"b' 'a\?b')");
  ASSERT_EQ(tokens.size(), 6u);

  // The quotes are part of the identifier.
  EXPECT_EQ(tokens[0].text, "'x'");
  EXPECT_EQ(tokens[1].text, "'+foo'");
  EXPECT_EQ(tokens[2].text, R"('13\'H')");

  // Redundant escapes are the same as the non-escaped characters.
  EXPECT_EQ(tokens[3].text, R"('a"b')");
  EXPECT_EQ(tokens[4].text, "'a?b'");
}

TEST(ModelicaLexer, integers) {
  auto tokens = tokenize("0 42 9223372036854775807");
  ASSERT_EQ(tokens.size(), 4u);

  EXPECT_EQ(tokens[0].kind, TokenKind::Integer);
  EXPECT_EQ(tokens[0].integerValue, 0);
  EXPECT_EQ(tokens[1].integerValue, 42);
  EXPECT_EQ(tokens[2].integerValue, 9223372036854775807);
}

TEST(ModelicaLexer, integerTooLarge) {
  auto tokens = tokenize("9223372036854775808");
  ASSERT_EQ(tokens.back().kind, TokenKind::Error);
  EXPECT_EQ(tokens.back().text, "integer literal is too large");
}

TEST(ModelicaLexer, reals) {
  auto tokens = tokenize("1.5 2. .5 1e3 1.e-3 2.5E+2");
  ASSERT_EQ(tokens.size(), 7u);

  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(tokens[i].kind, TokenKind::Real);
  }

  EXPECT_DOUBLE_EQ(tokens[0].realValue, 1.5);
  EXPECT_DOUBLE_EQ(tokens[1].realValue, 2);
  EXPECT_DOUBLE_EQ(tokens[2].realValue, 0.5);
  EXPECT_DOUBLE_EQ(tokens[3].realValue, 1000);
  EXPECT_DOUBLE_EQ(tokens[4].realValue, 0.001);
  EXPECT_DOUBLE_EQ(tokens[5].realValue, 250);
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

  EXPECT_EQ(tokens[0].kind, TokenKind::String);
  EXPECT_EQ(tokens[0].text, "abc");
  EXPECT_EQ(tokens[1].text, "a\"b\n");
  EXPECT_EQ(tokens[2].text, "");
}

TEST(ModelicaLexer, unterminatedString) {
  auto tokens = tokenize("\"abc");
  ASSERT_EQ(tokens.back().kind, TokenKind::Error);
  EXPECT_EQ(tokens.back().text, "unterminated string");
}

TEST(ModelicaLexer, invalidEscape) {
  auto tokens = tokenize(R"("a\qb")");
  ASSERT_EQ(tokens.back().kind, TokenKind::Error);
  EXPECT_EQ(tokens.back().text, "invalid escape sequence in string");
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
  ASSERT_EQ(tokens.back().kind, TokenKind::Error);
  EXPECT_EQ(tokens.back().text, "unterminated comment");
}

TEST(ModelicaLexer, unexpectedCharacter) {
  auto tokens = tokenize("a # b");
  ASSERT_EQ(tokens.back().kind, TokenKind::Error);
  EXPECT_EQ(tokens.back().text, "unexpected character '#'");
}

TEST(ModelicaLexer, locations) {
  auto tokens = tokenize("model M\n  Real xy;\nend M;");

  // 'xy'
  const Token &token = tokens[3];
  ASSERT_EQ(token.kind, TokenKind::Identifier);
  EXPECT_EQ(token.location.begin.line, 2);
  EXPECT_EQ(token.location.begin.column, 8);
  EXPECT_EQ(token.location.end.line, 2);
  EXPECT_EQ(token.location.end.column, 9);
}
