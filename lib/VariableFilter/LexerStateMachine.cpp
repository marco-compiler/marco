#include "marco/VariableFilter/LexerStateMachine.h"

using namespace ::marco;
using namespace ::marco::vf;

static bool isDigit(char c) { return ('0' <= c && c <= '9'); }

static bool isNonDigit(char c) {
  return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '_';
}

namespace marco::vf {
LexerStateMachine::LexerStateMachine(std::shared_ptr<SourceFile> file,
                                     char first)
    : state(State::Normal), current('\0'), next(first), identifier(""),
      regex(""), currentPosition(SourcePosition(file, 1, 0)),
      beginPosition(SourcePosition(file, 1, 0)),
      endPosition(SourcePosition(file, 1, 0)) {
  symbols['('] = TokenKind::LPar;
  symbols[')'] = TokenKind::RPar;
  symbols['['] = TokenKind::LSquare;
  symbols[']'] = TokenKind::RSquare;
  symbols[','] = TokenKind::Comma;
  symbols[';'] = TokenKind::Semicolons;
  symbols[':'] = TokenKind::Colons;
  symbols['$'] = TokenKind::Dollar;
  keywordMap["der"] = TokenKind::DerKeyword;
}

std::string LexerStateMachine::getIdentifier() const { return identifier; }

std::string LexerStateMachine::getRegex() const { return regex; }

int64_t LexerStateMachine::getInt() const { return numberLexer.get(); }

llvm::StringRef LexerStateMachine::getError() const { return error; }

SourcePosition LexerStateMachine::getCurrentPosition() const {
  return currentPosition;
}

SourceRange LexerStateMachine::getTokenPosition() const {
  return {beginPosition, endPosition};
}

void LexerStateMachine::advance(char c) {
  current = next;
  next = c;
  ++currentPosition.column;

  if (current == '\n') {
    currentPosition.column = 0;
    ++currentPosition.line;
  }
}

void LexerStateMachine::setTokenBeginPosition() {
  beginPosition = currentPosition;
}

void LexerStateMachine::setTokenEndPosition() { endPosition = currentPosition; }

Token LexerStateMachine::makeToken(TokenKind kind) {
  return Token(kind, SourceRange(beginPosition, endPosition));
}

TokenKind LexerStateMachine::stringToToken(llvm::StringRef str) const {
  if (auto iter = keywordMap.find(str); iter != keywordMap.end()) {
    return iter->getValue();
  }

  return TokenKind::Identifier;
}

TokenKind LexerStateMachine::charToToken(char c) const {
  if (auto iter = symbols.find(c); iter != symbols.end()) {
    return iter->second;
  }

  return TokenKind::Error;
}

template <LexerStateMachine::State s>
std::optional<Token> LexerStateMachine::scan() {
  return std::nullopt;
}

Token LexerStateMachine::trySymbolScan() {
  state = State::Normal;
  Token token = makeToken(charToToken(current));

  if (token.isa<TokenKind::Error>()) {
    error = "Unexpected character '";
    error.push_back(current);
    error.push_back('\'');
  }

  return token;
}

template <>
std::optional<Token>
LexerStateMachine::scan<LexerStateMachine::State::ParsingId>() {
  setTokenEndPosition();
  identifier.push_back(current);

  if (!isDigit(next) && !isNonDigit(next) && next != '.') {
    state = State::Normal;
    return makeToken(stringToToken(identifier), getIdentifier());
  }

  return std::nullopt;
}

template <>
std::optional<Token>
LexerStateMachine::scan<LexerStateMachine::State::ParsingNum>() {
  if (isDigit(current)) {
    setTokenEndPosition();
    numberLexer += (current - '0');
  }

  if (!isDigit(next)) {
    setTokenEndPosition();
    state = State::Normal;
    return makeToken(TokenKind::Integer, getInt());
  }

  return std::nullopt;
}

template <>
std::optional<Token>
LexerStateMachine::scan<LexerStateMachine::State::ParsingRegex>() {
  if (current == '/') {
    setTokenEndPosition();
    state = State::Normal;
    return makeToken(TokenKind::Regex, getRegex());
  }

  if (current == '\0') {
    setTokenEndPosition();
    state = State::End;
    error = "Reached end of string while parsing a regex";
    return makeToken(TokenKind::Error, getError());
  }

  regex.push_back(current);
  return std::nullopt;
}

template <>
std::optional<Token>
LexerStateMachine::scan<LexerStateMachine::State::Normal>() {
  if (std::isspace(current) != 0) {
    return std::nullopt;
  }

  setTokenBeginPosition();
  setTokenEndPosition();

  if (isNonDigit(current)) {
    state = State::ParsingId;
    identifier = "";

    return scan<State::ParsingId>();
  }

  if (isDigit(current)) {
    state = State::ParsingNum;
    numberLexer = IntegerLexer<10>();
    return scan<State::ParsingNum>();
  }

  if (current == '/') {
    state = State::ParsingRegex;
    regex = "";
    return std::nullopt;
  }

  if (current == '\0') {
    state = State::End;
    return makeToken(TokenKind::EndOfFile);
  }

  return trySymbolScan();
}

std::optional<Token> LexerStateMachine::step(char c) {
  advance(c);

  switch (state) {
  case (State::Normal):
    return scan<State::Normal>();

  case (State::ParsingNum):
    return scan<State::ParsingNum>();

  case (State::ParsingRegex):
    return scan<State::ParsingRegex>();

  case (State::ParsingId):
    return scan<State::ParsingId>();

  case (State::End):
    return makeToken(TokenKind::EndOfFile);

  case (State::IgnoreNextChar):
    state = State::Normal;
    return std::nullopt;
  }

  error = "Unhandled Lexer State";
  return makeToken(TokenKind::Error, getError());
}
} // namespace marco::vf
