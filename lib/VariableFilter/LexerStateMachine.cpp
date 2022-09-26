#include "marco/VariableFilter/LexerStateMachine.h"

using namespace ::marco;
using namespace ::marco::vf;

static bool isDigit(char c)
{
  return ('0' <= c && c <= '9');
}

static bool isNonDigit(char c)
{
  return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '_';
}

namespace marco::vf
{
  LexerStateMachine::LexerStateMachine(std::shared_ptr<SourceFile> file, char first)
      : state(State::Normal),
        current('\0'),
        next(first),
        lastIdentifier(""),
        lastRegex(""),
        currentPosition(SourcePosition(file, 1, 0)),
        beginPosition(SourcePosition(file, 1, 0)),
        endPosition(SourcePosition(file, 1, 0))
  {
    symbols['('] = Token::LPar;
    symbols[')'] = Token::RPar;
    symbols['['] = Token::LSquare;
    symbols[']'] = Token::RSquare;
    symbols[','] = Token::Comma;
    symbols[';'] = Token::Semicolons;
    symbols[':'] = Token::Colons;
    symbols['$'] = Token::Dollar;
    keywordMap["der"] = Token::DerKeyword;
  }

  const std::string& LexerStateMachine::getLastIdentifier() const
  {
    return lastIdentifier;
  }

  const std::string& LexerStateMachine::getLastRegex() const
  {
    return lastRegex;
  }

  long LexerStateMachine::getLastInt() const
  {
    return lastNum.get();
  }

  const std::string& LexerStateMachine::getLastError() const
  {
    return error;
  }

  SourcePosition LexerStateMachine::getCurrentPosition() const
  {
    return currentPosition;
  }

  SourceRange LexerStateMachine::getTokenPosition() const
  {
    return SourceRange(beginPosition, endPosition);
  }

  void LexerStateMachine::advance(char c)
  {
    current = next;
    next = c;
    ++currentPosition.column;

    if (current == '\n') {
      currentPosition.column = 0;
      ++currentPosition.line;
    }
  }

  void LexerStateMachine::setTokenBeginPosition()
  {
    beginPosition = currentPosition;
  }

  void LexerStateMachine::setTokenEndPosition()
  {
    endPosition = currentPosition;
  }

  Token LexerStateMachine::stringToToken(llvm::StringRef str) const
  {
    if (auto iter = keywordMap.find(str); iter != keywordMap.end()) {
      return iter->getValue();
    }

    return Token::Ident;
  }

  Token LexerStateMachine::charToToken(char c) const
  {
    if (auto iter = symbols.find(c); iter != symbols.end()) {
      return iter->second;
    }

    return Token::Error;
  }

  template<LexerStateMachine::State s>
  Token LexerStateMachine::scan() {
    return Token::None;
  }

  Token LexerStateMachine::tryScanSymbol()
  {
    state = State::Normal;
    Token token = charToToken(current);

    if (token == Token::Error) {
      error = "Unexpected character ";
      error.push_back(current);
    }

    return token;
  }

  template<>
  Token LexerStateMachine::scan<LexerStateMachine::State::ParsingId>()
  {
    lastIdentifier.push_back(current);

    if (!isDigit(next) && !isNonDigit(next) && next != '.') {
      state = State::Normal;
      setTokenEndPosition();
      return stringToToken(lastIdentifier);
    }

    return Token::None;
  }

  template<>
  Token LexerStateMachine::scan<LexerStateMachine::State::ParsingNum>()
  {
    if (isDigit(current)) {
      lastNum += (current - '0');
    }

    if (!isDigit(next)) {
      state = State::Normal;
      setTokenEndPosition();
      return Token::Integer;
    }

    return Token::None;
  }

  template<>
  Token LexerStateMachine::scan<LexerStateMachine::State::ParsingRegex>()
  {
    if (current == '/') {
      state = State::Normal;
      setTokenEndPosition();
      return Token::Regex;
    }

    if (current == '\0') {
      state = State::End;
      error = "Reached end of string while parsing a regex";
      return Token::Error;
    }

    lastRegex.push_back(current);
    return Token::None;
  }

  template<>
  Token LexerStateMachine::scan<LexerStateMachine::State::Normal>()
  {
    if (std::isspace(current) != 0) {
      return Token::None;
    }

    setTokenBeginPosition();
    setTokenEndPosition();

    if (isNonDigit(current)) {
      state = State::ParsingId;
      lastIdentifier = "";

      return scan<State::ParsingId>();
    }

    if (isDigit(current)) {
      state = State::ParsingNum;
      lastNum = IntegerLexer<10>();
      return scan<State::ParsingNum>();
    }

    if (current == '/') {
      state = State::ParsingRegex;
      lastRegex = "";
      return Token::None;
    }

    if (current == '\0') {
      state = State::End;
      return Token::EndOfFile;
    }

    return tryScanSymbol();
  }

  Token LexerStateMachine::step(char c)
  {
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
        return Token::EndOfFile;

      case (State::IgnoreNextChar):
        state = State::Normal;
        return Token::None;
    }

    error = "Unhandled Lexer State";
    return Token::Error;
  }
}
