#include "marco/Parser/Modelica/StateMachine.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdlib>
#include <limits>

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
bool isIdentifierCharacter(char c) { return isNonDigit(c) || isDigit(c); }

bool isWhitespace(char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' ||
         c == '\v';
}
} // namespace

namespace marco::parser::modelica {
StateMachine::StateMachine(const std::shared_ptr<SourceFile> &file, char first)
    : next(first), currentPosition(file, 1, 0), nextPosition(file, 1, 1),
      tokenBegin(file, 1, 1) {
  // The byte order mark, if any, is skipped.
  if (llvm::StringRef(file->getBuffer()->getBuffer())
          .starts_with("\xEF\xBB\xBF")) {
    byteOrderMark = "\xEF\xBB\xBF";
  }
}

std::optional<Token> StateMachine::step(char c) {
  current = next;
  next = c;
  currentPosition = nextPosition;

  if (!byteOrderMark.empty()) {
    byteOrderMark = byteOrderMark.drop_front();
    return std::nullopt;
  }

  if (current == '\n') {
    ++nextPosition.line;
    nextPosition.column = 1;
  } else if (current != '\0') {
    ++nextPosition.column;
  }

  switch (state) {
  case State::Normal:
    return scanNormal();

  case State::SingleLineComment:
    if (current == '\0') {
      state = State::Normal;
      beginToken();
      return makeToken(TokenKind::EndOfFile);
    }

    if (current == '\n') {
      state = State::Normal;
    }

    return std::nullopt;

  case State::MultiLineCommentOpening:
    state = State::MultiLineComment;
    return std::nullopt;

  case State::MultiLineComment:
    if (current == '\0') {
      state = State::Normal;
      return makeError("unterminated comment");
    }

    if (current == '*' && next == '/') {
      state = State::MultiLineCommentClosing;
    }

    return std::nullopt;

  case State::MultiLineCommentClosing:
    state = State::Normal;
    return std::nullopt;

  case State::Identifier:
    text += current;

    if (isIdentifierCharacter(next)) {
      return std::nullopt;
    }

    state = State::Normal;

    if (auto keyword = getKeyword(text)) {
      return makeToken(*keyword);
    }

    return makeToken(TokenKind::Identifier);

  case State::QuotedIdentifier:
    if (current == '\0' || current == '\n' || current == '\r') {
      state = State::Normal;
      return makeError("unterminated quoted identifier");
    }

    if (current == '\\') {
      state = State::QuotedIdentifierEscape;
      return std::nullopt;
    }

    text += current;

    if (current != '\'') {
      return std::nullopt;
    }

    state = State::Normal;

    if (text == "''") {
      return makeError("empty quoted identifier");
    }

    return makeToken(TokenKind::Identifier);

  case State::QuotedIdentifierEscape:
    if (!getEscapedChar(current)) {
      state = State::Normal;
      return makeError("invalid escape sequence in quoted identifier");
    }

    // The redundant escapes are the same as the non-escaped characters,
    // while the other escapes are kept as written.
    if (current != '?' && current != '"') {
      text += '\\';
    }

    text += current;
    state = State::QuotedIdentifier;
    return std::nullopt;

  case State::String:
    if (current == '\0') {
      state = State::Normal;
      return makeError("unterminated string");
    }

    if (current == '\\') {
      state = State::StringEscape;
      return std::nullopt;
    }

    if (current == '"') {
      state = State::Normal;
      return makeToken(TokenKind::String);
    }

    text += current;
    return std::nullopt;

  case State::StringEscape:
    if (auto escaped = getEscapedChar(current)) {
      text += *escaped;
      state = State::String;
      return std::nullopt;
    }

    state = State::Normal;
    return makeError("invalid escape sequence in string");

  case State::NumberInteger:
  case State::NumberFraction:
  case State::NumberExponentMarker:
  case State::NumberExponentSign:
  case State::NumberExponent:
    text += current;
    return continueNumber();

  case State::SymbolSecondCharacter:
    state = State::Normal;
    return makeToken(symbolKind);
  }

  llvm_unreachable("Unknown state");
  return std::nullopt;
}

std::optional<Token> StateMachine::scanNormal() {
  if (current == '\0') {
    beginToken();
    return makeToken(TokenKind::EndOfFile);
  }

  if (isWhitespace(current)) {
    return std::nullopt;
  }

  if (current == '/' && next == '/') {
    state = State::SingleLineComment;
    return std::nullopt;
  }

  if (current == '/' && next == '*') {
    beginToken();
    state = State::MultiLineCommentOpening;
    return std::nullopt;
  }

  beginToken();

  if (isNonDigit(current)) {
    text += current;

    if (isIdentifierCharacter(next)) {
      state = State::Identifier;
      return std::nullopt;
    }

    if (auto keyword = getKeyword(text)) {
      return makeToken(*keyword);
    }

    return makeToken(TokenKind::Identifier);
  }

  if (current == '\'') {
    text += current;
    state = State::QuotedIdentifier;
    return std::nullopt;
  }

  if (isDigit(current)) {
    text += current;
    state = State::NumberInteger;
    return continueNumber();
  }

  if (current == '.' && isDigit(next)) {
    text += current;
    isReal = true;
    state = State::NumberFraction;
    return continueNumber();
  }

  if (current == '"') {
    state = State::String;
    return std::nullopt;
  }

  return scanSymbol();
}

std::optional<Token> StateMachine::scanSymbol() {
  auto twoCharacters = [&](TokenKind kind) -> std::optional<Token> {
    symbolKind = kind;
    state = State::SymbolSecondCharacter;
    return std::nullopt;
  };

  switch (current) {
  case '+':
    return makeToken(TokenKind::Plus);

  case '-':
    return makeToken(TokenKind::Minus);

  case '*':
    return makeToken(TokenKind::Star);

  case '/':
    return makeToken(TokenKind::Slash);

  case '^':
    return makeToken(TokenKind::Caret);

  case ',':
    return makeToken(TokenKind::Comma);

  case ';':
    return makeToken(TokenKind::Semicolon);

  case '(':
    return makeToken(TokenKind::LParen);

  case ')':
    return makeToken(TokenKind::RParen);

  case '[':
    return makeToken(TokenKind::LSquare);

  case ']':
    return makeToken(TokenKind::RSquare);

  case '{':
    return makeToken(TokenKind::LCurly);

  case '}':
    return makeToken(TokenKind::RCurly);

  case '.':
    switch (next) {
    case '+':
      return twoCharacters(TokenKind::PlusEW);

    case '-':
      return twoCharacters(TokenKind::MinusEW);

    case '*':
      return twoCharacters(TokenKind::StarEW);

    case '/':
      return twoCharacters(TokenKind::SlashEW);

    case '^':
      return twoCharacters(TokenKind::CaretEW);

    default:
      return makeToken(TokenKind::Dot);
    }

  case '=':
    return next == '=' ? twoCharacters(TokenKind::EqualEqual)
                       : makeToken(TokenKind::Equal);

  case '<':
    if (next == '=') {
      return twoCharacters(TokenKind::LessEqual);
    }

    if (next == '>') {
      return twoCharacters(TokenKind::NotEqual);
    }

    return makeToken(TokenKind::Less);

  case '>':
    return next == '=' ? twoCharacters(TokenKind::GreaterEqual)
                       : makeToken(TokenKind::Greater);

  case ':':
    return next == '=' ? twoCharacters(TokenKind::Assignment)
                       : makeToken(TokenKind::Colon);

  default:
    return makeError("unexpected character '" + std::string(1, current) + "'");
  }
}

std::optional<Token> StateMachine::continueNumber() {
  bool isExponentMarker = next == 'e' || next == 'E';

  switch (state) {
  case State::NumberInteger:
    if (isDigit(next)) {
      return std::nullopt;
    }

    // A trailing dot makes the number a Real one, as in '2.'.
    if (next == '.') {
      isReal = true;
      state = State::NumberFraction;
      return std::nullopt;
    }

    if (isExponentMarker) {
      state = State::NumberExponentMarker;
      return std::nullopt;
    }

    break;

  case State::NumberFraction:
    if (isDigit(next)) {
      return std::nullopt;
    }

    if (isExponentMarker) {
      state = State::NumberExponentMarker;
      return std::nullopt;
    }

    break;

  case State::NumberExponentMarker:
    isReal = true;

    if (next == '+' || next == '-') {
      state = State::NumberExponentSign;
      return std::nullopt;
    }

    [[fallthrough]];

  case State::NumberExponentSign:
    if (isDigit(next)) {
      state = State::NumberExponent;
      return std::nullopt;
    }

    state = State::Normal;
    return makeError("missing digits in the exponent");

  case State::NumberExponent:
    if (isDigit(next)) {
      return std::nullopt;
    }

    break;

  default:
    llvm_unreachable("Not scanning a number");
  }

  state = State::Normal;
  return makeNumber();
}

void StateMachine::beginToken() {
  tokenBegin = currentPosition;
  text.clear();
  isReal = false;
}

Token StateMachine::makeToken(TokenKind kind) const {
  SourceRange location(tokenBegin, currentPosition);

  if (kind == TokenKind::Identifier || kind == TokenKind::String) {
    return {kind, location, text};
  }

  return Token(kind, location);
}

Token StateMachine::makeError(llvm::StringRef message) const {
  return {TokenKind::Error, SourceRange(tokenBegin, currentPosition), message};
}

Token StateMachine::makeNumber() const {
  SourceRange location(tokenBegin, currentPosition);

  if (isReal) {
    return {TokenKind::Real, location, std::strtod(text.c_str(), nullptr)};
  }

  uint64_t value;

  if (llvm::StringRef(text).getAsInteger(10, value) ||
      value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return makeError("integer literal is too large");
  }

  return {TokenKind::Integer, location, static_cast<int64_t>(value)};
}
} // namespace marco::parser::modelica
