#include "marco/Parser/Modelica/Lexer.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include <cstdlib>

using namespace ::marco;
using namespace ::marco::parser::modelica;

namespace {
bool isDigit(char c) { return c >= '0' && c <= '9'; }

bool isNonDigit(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

std::optional<TokenKind> getKeyword(llvm::StringRef identifier) {
  static const llvm::StringMap<TokenKind> keywords = {
      {"algorithm", TokenKind::Algorithm},
      {"and", TokenKind::And},
      {"annotation", TokenKind::Annotation},
      {"block", TokenKind::Block},
      {"break", TokenKind::Break},
      {"class", TokenKind::Class},
      {"connect", TokenKind::Connect},
      {"connector", TokenKind::Connector},
      {"constant", TokenKind::Constant},
      {"constrainedby", TokenKind::ConstrainedBy},
      {"der", TokenKind::Der},
      {"discrete", TokenKind::Discrete},
      {"each", TokenKind::Each},
      {"else", TokenKind::Else},
      {"elseif", TokenKind::ElseIf},
      {"elsewhen", TokenKind::ElseWhen},
      {"encapsulated", TokenKind::Encapsulated},
      {"end", TokenKind::End},
      {"enumeration", TokenKind::Enumeration},
      {"equation", TokenKind::Equation},
      {"expandable", TokenKind::Expandable},
      {"extends", TokenKind::Extends},
      {"external", TokenKind::External},
      {"false", TokenKind::False},
      {"final", TokenKind::Final},
      {"flow", TokenKind::Flow},
      {"for", TokenKind::For},
      {"function", TokenKind::Function},
      {"if", TokenKind::If},
      {"import", TokenKind::Import},
      {"impure", TokenKind::Impure},
      {"in", TokenKind::In},
      {"initial", TokenKind::Initial},
      {"inner", TokenKind::Inner},
      {"input", TokenKind::Input},
      {"loop", TokenKind::Loop},
      {"model", TokenKind::Model},
      {"not", TokenKind::Not},
      {"operator", TokenKind::Operator},
      {"or", TokenKind::Or},
      {"outer", TokenKind::Outer},
      {"output", TokenKind::Output},
      {"package", TokenKind::Package},
      {"parameter", TokenKind::Parameter},
      {"partial", TokenKind::Partial},
      {"protected", TokenKind::Protected},
      {"public", TokenKind::Public},
      {"pure", TokenKind::Pure},
      {"record", TokenKind::Record},
      {"redeclare", TokenKind::Redeclare},
      {"replaceable", TokenKind::Replaceable},
      {"return", TokenKind::Return},
      {"stream", TokenKind::Stream},
      {"then", TokenKind::Then},
      {"time", TokenKind::Time},
      {"true", TokenKind::True},
      {"type", TokenKind::Type},
      {"when", TokenKind::When},
      {"while", TokenKind::While},
      {"within", TokenKind::Within}};

  auto it = keywords.find(identifier);

  if (it == keywords.end()) {
    return std::nullopt;
  }

  return it->second;
}

/// Get the character represented by an escape sequence.
std::optional<char> getEscapedChar(char c) {
  switch (c) {
  case '\'':
  case '"':
  case '?':
  case '\\':
    return c;
  case 'a':
    return '\a';
  case 'b':
    return '\b';
  case 'f':
    return '\f';
  case 'n':
    return '\n';
  case 'r':
    return '\r';
  case 't':
    return '\t';
  case 'v':
    return '\v';
  default:
    return std::nullopt;
  }
}
} // namespace

namespace marco::parser::modelica {
Lexer::Lexer(std::shared_ptr<SourceFile> file)
    : file(file), buffer(file->getBuffer()->getBuffer()),
      lastPosition(file, 1, 0) {
  // Skip the byte order mark, if any.
  if (buffer.starts_with("\xEF\xBB\xBF")) {
    index = 3;
  }
}

char Lexer::peek(size_t offset) const {
  if (index + offset >= buffer.size()) {
    return '\0';
  }

  return buffer[index + offset];
}

void Lexer::advance() {
  if (index >= buffer.size()) {
    return;
  }

  lastPosition = position();

  if (buffer[index] == '\n') {
    ++line;
    column = 1;
  } else {
    ++column;
  }

  ++index;
}

SourcePosition Lexer::position() const {
  return SourcePosition(file, line, column);
}

Token Lexer::makeToken(TokenKind kind, SourcePosition begin) const {
  Token token;
  token.kind = kind;
  token.location = SourceRange(std::move(begin), lastPosition);
  return token;
}

Token Lexer::makeError(SourcePosition begin, llvm::StringRef message) const {
  Token token = makeToken(TokenKind::Error, std::move(begin));
  token.text = message.str();
  return token;
}

bool Lexer::skipWhitespaceAndComments(Token &error) {
  while (true) {
    char c = peek();

    if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' ||
        c == '\v') {
      advance();
      continue;
    }

    if (c == '/' && peek(1) == '/') {
      while (peek() != '\n' && peek() != '\0') {
        advance();
      }

      continue;
    }

    if (c == '/' && peek(1) == '*') {
      SourcePosition begin = position();
      advance();
      advance();

      while (!(peek() == '*' && peek(1) == '/')) {
        if (index >= buffer.size()) {
          error = makeError(begin, "unterminated comment");
          return false;
        }

        advance();
      }

      advance();
      advance();
      continue;
    }

    return true;
  }
}

Token Lexer::next() {
  Token error;

  if (!skipWhitespaceAndComments(error)) {
    return error;
  }

  if (index >= buffer.size() || peek() == '\0') {
    Token token;
    token.kind = TokenKind::EndOfFile;
    token.location = SourceRange(position(), position());
    return token;
  }

  char c = peek();

  if (isNonDigit(c)) {
    return scanIdentifier();
  }

  if (c == '\'') {
    return scanQuotedIdentifier();
  }

  if (isDigit(c) || (c == '.' && isDigit(peek(1)))) {
    return scanNumber();
  }

  if (c == '"') {
    return scanString();
  }

  return scanSymbol();
}

std::vector<Token> Lexer::tokenize() {
  std::vector<Token> tokens;

  do {
    tokens.push_back(next());
  } while (!tokens.back().is(TokenKind::EndOfFile) &&
           !tokens.back().is(TokenKind::Error));

  return tokens;
}

Token Lexer::scanIdentifier() {
  SourcePosition begin = position();
  size_t start = index;

  while (isNonDigit(peek()) || isDigit(peek())) {
    advance();
  }

  llvm::StringRef identifier = buffer.slice(start, index);

  if (auto keyword = getKeyword(identifier)) {
    return makeToken(*keyword, begin);
  }

  Token token = makeToken(TokenKind::Identifier, begin);
  token.text = identifier.str();
  return token;
}

Token Lexer::scanQuotedIdentifier() {
  SourcePosition begin = position();
  std::string text = "'";
  advance();

  while (peek() != '\'') {
    char c = peek();

    if (c == '\0' || c == '\n' || c == '\r') {
      return makeError(begin, "unterminated quoted identifier");
    }

    if (c == '\\') {
      char escaped = peek(1);

      if (!getEscapedChar(escaped)) {
        advance();
        return makeError(begin, "invalid escape sequence in quoted identifier");
      }

      advance();
      advance();

      // The redundant escapes are the same as the non-escaped characters,
      // while the other escapes are kept as written.
      if (escaped == '?' || escaped == '"') {
        text += escaped;
      } else {
        text += '\\';
        text += escaped;
      }

      continue;
    }

    text += c;
    advance();
  }

  advance();
  text += "'";

  if (text == "''") {
    return makeError(begin, "empty quoted identifier");
  }

  Token token = makeToken(TokenKind::Identifier, begin);
  token.text = std::move(text);
  return token;
}

Token Lexer::scanNumber() {
  SourcePosition begin = position();
  size_t start = index;
  bool isReal = false;

  while (isDigit(peek())) {
    advance();
  }

  // A trailing dot makes the number a Real one, as in '2.'.
  if (peek() == '.') {
    isReal = true;
    advance();

    while (isDigit(peek())) {
      advance();
    }
  }

  if (peek() == 'e' || peek() == 'E') {
    size_t offset = (peek(1) == '+' || peek(1) == '-') ? 2 : 1;

    if (isDigit(peek(offset))) {
      isReal = true;

      for (size_t i = 0; i < offset; ++i) {
        advance();
      }

      while (isDigit(peek())) {
        advance();
      }
    }
  }

  llvm::StringRef text = buffer.slice(start, index);

  if (isReal) {
    Token token = makeToken(TokenKind::Real, begin);
    token.realValue = std::strtod(text.str().c_str(), nullptr);
    return token;
  }

  uint64_t value;

  if (text.getAsInteger(10, value) ||
      value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return makeError(begin, "integer literal is too large");
  }

  Token token = makeToken(TokenKind::Integer, begin);
  token.integerValue = static_cast<int64_t>(value);
  return token;
}

Token Lexer::scanString() {
  SourcePosition begin = position();
  std::string text;
  advance();

  while (peek() != '"') {
    char c = peek();

    if (index >= buffer.size()) {
      return makeError(begin, "unterminated string");
    }

    if (c == '\\') {
      auto escaped = getEscapedChar(peek(1));

      if (!escaped) {
        advance();
        return makeError(begin, "invalid escape sequence in string");
      }

      advance();
      advance();
      text += *escaped;
      continue;
    }

    text += c;
    advance();
  }

  advance();
  Token token = makeToken(TokenKind::String, begin);
  token.text = std::move(text);
  return token;
}

Token Lexer::scanSymbol() {
  SourcePosition begin = position();
  char c = peek();
  char next = peek(1);

  auto single = [&](TokenKind kind) {
    advance();
    return makeToken(kind, begin);
  };

  auto twoChars = [&](TokenKind kind) {
    advance();
    advance();
    return makeToken(kind, begin);
  };

  switch (c) {
  case '+':
    return single(TokenKind::Plus);
  case '-':
    return single(TokenKind::Minus);
  case '*':
    return single(TokenKind::Star);
  case '/':
    return single(TokenKind::Slash);
  case '^':
    return single(TokenKind::Caret);
  case ',':
    return single(TokenKind::Comma);
  case ';':
    return single(TokenKind::Semicolon);
  case '(':
    return single(TokenKind::LParen);
  case ')':
    return single(TokenKind::RParen);
  case '[':
    return single(TokenKind::LSquare);
  case ']':
    return single(TokenKind::RSquare);
  case '{':
    return single(TokenKind::LCurly);
  case '}':
    return single(TokenKind::RCurly);

  case '.':
    switch (next) {
    case '+':
      return twoChars(TokenKind::PlusEW);
    case '-':
      return twoChars(TokenKind::MinusEW);
    case '*':
      return twoChars(TokenKind::StarEW);
    case '/':
      return twoChars(TokenKind::SlashEW);
    case '^':
      return twoChars(TokenKind::CaretEW);
    default:
      return single(TokenKind::Dot);
    }

  case '=':
    return next == '=' ? twoChars(TokenKind::EqualEqual)
                       : single(TokenKind::Equal);

  case '<':
    if (next == '=') {
      return twoChars(TokenKind::LessEqual);
    }

    if (next == '>') {
      return twoChars(TokenKind::NotEqual);
    }

    return single(TokenKind::Less);

  case '>':
    return next == '=' ? twoChars(TokenKind::GreaterEqual)
                       : single(TokenKind::Greater);

  case ':':
    return next == '=' ? twoChars(TokenKind::Assignment)
                       : single(TokenKind::Colon);

  default:
    advance();
    return makeError(begin, "unexpected character '" + std::string(1, c) +
                                "'");
  }
}
} // namespace marco::parser::modelica
