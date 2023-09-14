#include "marco/Parser/ModelicaStateMachine.h"

using namespace ::marco;
using namespace ::marco::parser;

static bool isDigit(char c)
{
  return ('0' <= c && c <= '9');
}

static bool isNonDigit(char c)
{
  return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '_';
}

/// Get the escaped version of a character, but ony if that escaped version has
/// a special meaning in the ASCII table.
/// For example, the escaped version of 'n' is a new line, but the escaped
/// version of 'z' means nothing special and thus only 'z' is returned.
static char escapedChar(char c)
{
  constexpr int alert = 7;
  constexpr int backspace = 8;
  constexpr int fromFeed = 12;

  switch (c) {
    case 'a':
      return static_cast<char>(alert);

    case 'b':
      return static_cast<char>(backspace);

    case 'f':
      return static_cast<char>(fromFeed);

    case 'n':
      return '\n';

    case 'r':
      return '\r';

    case 't':
      return '\t';

    case 'v':
      return '\v';

    default:
      return c;
  }
}

namespace marco::parser
{
  ModelicaStateMachine::ModelicaStateMachine(std::shared_ptr<SourceFile> file, char first)
    : state(State::Normal),
      current('\0'),
      next(first),
      identifier(""),
      stringValue(""),
      currentPosition(SourcePosition(file, 1, 0)),
      beginPosition(SourcePosition(file, 1, 0)),
      endPosition(SourcePosition(file, 1, 0))
  {
    // Populate the reserved keywords map
    reservedKeywords["algorithm"] = TokenKind::Algorithm;
    reservedKeywords["and"] = TokenKind::And;
    reservedKeywords["annotation"] = TokenKind::Annotation;
    reservedKeywords["block"] = TokenKind::Block;
    reservedKeywords["break"] = TokenKind::Break;
    reservedKeywords["class"] = TokenKind::Class;
    reservedKeywords["connect"] = TokenKind::Connect;
    reservedKeywords["connector"] = TokenKind::Connector;
    reservedKeywords["constant"] = TokenKind::Constant;
    reservedKeywords["constrainedby"] = TokenKind::ConstrainedBy;
    reservedKeywords["der"] = TokenKind::Der;
    reservedKeywords["discrete"] = TokenKind::Discrete;
    reservedKeywords["each"] = TokenKind::Each;
    reservedKeywords["else"] = TokenKind::Else;
    reservedKeywords["elseif"] = TokenKind::ElseIf;
    reservedKeywords["elsewhen"] = TokenKind::ElseWhen;
    reservedKeywords["encapsulated"] = TokenKind::Encapsulated;
    reservedKeywords["end"] = TokenKind::End;
    reservedKeywords["enumeration"] = TokenKind::Enumeration;
    reservedKeywords["equation"] = TokenKind::Equation;
    reservedKeywords["expandable"] = TokenKind::Expandable;
    reservedKeywords["extends"] = TokenKind::Extends;
    reservedKeywords["external"] = TokenKind::External;
    reservedKeywords["false"] = TokenKind::False;
    reservedKeywords["final"] = TokenKind::Final;
    reservedKeywords["flow"] = TokenKind::Flow;
    reservedKeywords["for"] = TokenKind::For;
    reservedKeywords["function"] = TokenKind::Function;
    reservedKeywords["if"] = TokenKind::If;
    reservedKeywords["import"] = TokenKind::Import;
    reservedKeywords["impure"] = TokenKind::Impure;
    reservedKeywords["in"] = TokenKind::In;
    reservedKeywords["initial"] = TokenKind::Initial;
    reservedKeywords["inner"] = TokenKind::Inner;
    reservedKeywords["input"] = TokenKind::Input;
    reservedKeywords["loop"] = TokenKind::Loop;
    reservedKeywords["model"] = TokenKind::Model;
    reservedKeywords["not"] = TokenKind::Not;
    reservedKeywords["operator"] = TokenKind::Operator;
    reservedKeywords["or"] = TokenKind::Or;
    reservedKeywords["outer"] = TokenKind::Outer;
    reservedKeywords["output"] = TokenKind::Output;
    reservedKeywords["package"] = TokenKind::Package;
    reservedKeywords["parameter"] = TokenKind::Parameter;
    reservedKeywords["partial"] = TokenKind::Partial;
    reservedKeywords["protected"] = TokenKind::Protected;
    reservedKeywords["public"] = TokenKind::Public;
    reservedKeywords["pure"] = TokenKind::Pure;
    reservedKeywords["record"] = TokenKind::Record;
    reservedKeywords["redeclare"] = TokenKind::Redeclare;
    reservedKeywords["replaceable"] = TokenKind::Replaceable;
    reservedKeywords["return"] = TokenKind::Return;
    reservedKeywords["stream"] = TokenKind::Stream;
    reservedKeywords["then"] = TokenKind::Then;
    reservedKeywords["true"] = TokenKind::True;
    reservedKeywords["type"] = TokenKind::Type;
    reservedKeywords["when"] = TokenKind::When;
    reservedKeywords["while"] = TokenKind::While;
    reservedKeywords["within"] = TokenKind::Within;
  }

  std::string ModelicaStateMachine::getIdentifier() const
  {
    return identifier;
  }

  std::string ModelicaStateMachine::getString() const
  {
    return stringValue;
  }

  int64_t ModelicaStateMachine::getInt() const
  {
    return numberLexer.getUpperPart();
  }

  double ModelicaStateMachine::getFloat() const
  {
    return numberLexer.get();
  }

  llvm::StringRef ModelicaStateMachine::getError() const
  {
    return error;
  }

  SourcePosition ModelicaStateMachine::getCurrentPosition() const
  {
    return currentPosition;
  }

  SourceRange ModelicaStateMachine::getTokenPosition() const
  {
    return {beginPosition, endPosition};
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingIdentifier>()
  {
    setTokenEndPosition();
    identifier.push_back(current);

    if (!isDigit(next) && !isNonDigit(next)) {
      state = State::Normal;

      if (auto it = reservedKeywords.find(identifier); it != reservedKeywords.end()) {
        return makeToken(it->second);
      }

      return makeToken(TokenKind::Identifier, getIdentifier());
    }

    return std::nullopt;
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingQIdentifier>()
  {
    setTokenEndPosition();

    if (current == '\\') {
      state = State::ParsingQIdentifierBackSlash;
      return std::nullopt;
    }

    if (current == '\'') {
      state = State::Normal;
      return makeToken(TokenKind::Identifier, getIdentifier());
    }

    if (next == '\0') {
      state = State::Normal;
      error = "unexpected end of file while parsing a q-identifier";
      return makeToken(TokenKind::Error, getError());
    }

    identifier.push_back(current);
    return std::nullopt;
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingNumber>()
  {
    setTokenEndPosition();

    if (isDigit(current)) {
      numberLexer.addUpper(current - '0');
    }

    if (current == '.') {
      state = State::ParsingFloat;
      return std::nullopt;
    }

    if (current == 'E' || current == 'e') {
      if (next == '+' || next == '-') {
        state = State::ParsingFloatExponentSign;
        return std::nullopt;
      }

      if (isDigit(next)) {
        numberLexer.setSign(true);
        state = State::ParsingFloatExponent;
        return std::nullopt;
      }

      state = State::Normal;
      error = "missing exponent in floating point number";
      return makeToken(TokenKind::Error, getError());
    }

    auto isAcceptable = [](char c) {
      return isDigit(c) || c == '.' || c == 'e' || c == 'E';
    };

    if (!isAcceptable(next)) {
      state = State::Normal;
      return makeToken(TokenKind::Integer, getInt());
    }

    return std::nullopt;
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingFloat>()
  {
    if (std::isspace(current) || current == '\0') {
      state = State::Normal;
      return makeToken(TokenKind::FloatingPoint, getFloat());
    }

    if (isDigit(current)) {
      numberLexer.addLower(current - '0');
    }

    if (current == 'E' || current == 'e') {
      if (next == '+' || next == '-') {
        state = State::ParsingFloatExponentSign;
        return std::nullopt;
      }

      if (isDigit(next)) {
        numberLexer.setSign(true);
        state = State::ParsingFloatExponent;
        return std::nullopt;
      }
    }

    if (isDigit(next) || next == 'E' || next == 'e') {
      return std::nullopt;
    }

    state = State::Normal;
    setTokenEndPosition();
    return makeToken(TokenKind::FloatingPoint, getFloat());
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingFloatExponentSign>()
  {
    if (std::isspace(current) || current == '\0') {
      state = State::Normal;
      error = "unexpected termination of the floating point number";
      return makeToken(TokenKind::Error, getError());
    }

    setTokenEndPosition();

    if (current == '-' || current == '+') {
      if (!isDigit(next)) {
        error = "Exponent sign must be followed by a number";
        state = State::Normal;
        return makeToken(TokenKind::Error, getError());
      }

      state = State::ParsingFloatExponent;
      numberLexer.setSign(current == '+');
      return std::nullopt;
    }

    error = "unexpected character '" + std::to_string(current) + "' in floating pointer number scan";
    state = State::Normal;
    return makeToken(TokenKind::Error, getError());
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingFloatExponent>()
  {
    if (std::isspace(current) || current == '\0') {
      state = State::Normal;
      //error = "unexpected termination of the floating point number";
      return makeToken(TokenKind::FloatingPoint, getFloat());
    }

    setTokenEndPosition();

    if (isDigit(current)) {
      numberLexer.addExponential(current - '0');
    }

    if (isDigit(next)) {
      return std::nullopt;
    }

    state = State::Normal;
    return makeToken(TokenKind::FloatingPoint, getFloat());
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingString>()
  {
    if (current == '\0') {
      state = State::End;
      error = "reached end of file while parsing a string";
      return makeToken(TokenKind::Error, getError());
    }

    setTokenEndPosition();

    if (current == '"') {
      state = State::Normal;
      return makeToken(TokenKind::String, getString());
    }

    if (current == '\\') {
      state = State::ParsingBackSlash;
      return std::nullopt;
    }

    stringValue.push_back(current);
    return std::nullopt;
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingSingleLineComment>()
  {
    setTokenEndPosition();

    if (next == '\0') {
      state = State::Normal;
    }

    if (current == '\n') {
      state = State::Normal;
    }

    return std::nullopt;
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingMultiLineComment>()
  {
    setTokenEndPosition();

    if (next == '\0') {
      state = State::Normal;
    }

    if (current == '*' && next == '/') {
      state = State::IgnoredCharacter;
    }

    return std::nullopt;
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingBackSlash>()
  {
    setTokenEndPosition();
    stringValue.push_back(escapedChar(current));
    state = State::ParsingString;
    return std::nullopt;
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingQIdentifierBackSlash>()
  {
    setTokenEndPosition();
    identifier.push_back(escapedChar(current));
    state = State::ParsingQIdentifier;
    return std::nullopt;
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingElementWiseSymbol>()
  {
    setTokenEndPosition();
    state = State::Normal;

    if (current == '+') {
      return makeToken(TokenKind::PlusEW);
    }

    if (current == '-') {
      return makeToken(TokenKind::MinusEW);
    }

    if (current == '*') {
      return makeToken(TokenKind::ProductEW);
    }

    if (current == '/') {
      return makeToken(TokenKind::DivisionEW);
    }

    if (current == '^') {
      return makeToken(TokenKind::PowEW);
    }

    error = "unexpected character '" + std::to_string(current) + "'";
    return makeToken(TokenKind::Error, getError());
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingEqualSymbol>()
  {
    setTokenEndPosition();
    state = State::Normal;

    if (current == '=') {
      return makeToken(TokenKind::Equal);
    }

    error = "unexpected character '" + std::to_string(current) + "'";
    return makeToken(TokenKind::Error, getError());
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingLessSymbol>()
  {
    setTokenEndPosition();
    state = State::Normal;

    if (current == '>') {
      return makeToken(TokenKind::NotEqual);
    }

    if (current == '=') {
      return makeToken(TokenKind::LessEqual);
    }

    error = "unexpected character '" + std::to_string(current) + "'";
    return makeToken(TokenKind::Error, getError());
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingGreaterSymbol>()
  {
    setTokenEndPosition();
    state = State::Normal;

    if (current == '=') {
      return makeToken(TokenKind::GreaterEqual);
    }

    error = "unexpected character '" + std::to_string(current) + "'";
    return makeToken(TokenKind::Error, getError());
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::ParsingColonSymbol>()
  {
    setTokenEndPosition();
    state = State::Normal;

    if (current == '=') {
      return makeToken(TokenKind::AssignmentOperator);
    }

    error = "unexpected character '" + std::to_string(current) + "'";
    return makeToken(TokenKind::Error, getError());
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::IgnoredCharacter>()
  {
    setTokenEndPosition();
    state = State::Normal;
    return std::nullopt;
  }

  template<>
  std::optional<Token> ModelicaStateMachine::scan<
      ModelicaStateMachine::State::Normal>()
  {
    if (std::isspace(current) != 0) {
      // Skip spaces
      return std::nullopt;
    }

    setTokenBeginPosition();
    setTokenEndPosition();

    if (isNonDigit(current) || (current == '$' && isNonDigit(next))) {
      state = State::ParsingIdentifier;
      identifier = "";

      return scan<State::ParsingIdentifier>();
    }

    if (current == '\'') {
      state = State::ParsingQIdentifier;
      identifier = "";
      return std::nullopt;
    }

    if (isDigit(current)) {
      state = State::ParsingNumber;
      numberLexer.reset();
      return scan<State::ParsingNumber>();
    }

    if (current == '"') {
      state = State::ParsingString;
      stringValue = "";
      return std::nullopt;
    }

    if (current == '/' && next == '/') {
      state = State::ParsingSingleLineComment;
      return std::nullopt;
    }

    if (current == '/' && next == '*') {
      state = State::ParsingMultiLineComment;
      return std::nullopt;
    }

    if (current == '\0') {
      state = State::End;
      return makeToken(TokenKind::EndOfFile);
    }

    return trySymbolScan();
  }

  std::optional<Token> ModelicaStateMachine::trySymbolScan()
  {
    if (current == '.') {
      switch (next) {
        case '+':
        case '-':
        case '*':
        case '/':
        case '^':
          state = State::ParsingElementWiseSymbol;
          return std::nullopt;
      }
    }

    if (current == '=' && next == '=') {
      state = State::ParsingEqualSymbol;
      return std::nullopt;
    }

    if (current == '<') {
      switch (next) {
        case '>':
        case '=':
          state = State::ParsingLessSymbol;
          return std::nullopt;
      }
    }

    if (current == '>' && next == '=') {
      state = State::ParsingGreaterSymbol;
      return std::nullopt;
    }

    if (current == ':' && next == '=') {
      state = State::ParsingColonSymbol;
      return std::nullopt;
    }

    state = State::Normal;

    switch (current) {
      case '+':
        return makeToken(TokenKind::Plus);

      case '-':
        return makeToken(TokenKind::Minus);

      case '*':
        return makeToken(TokenKind::Product);

      case '/':
        return makeToken(TokenKind::Division);

      case '^':
        return makeToken(TokenKind::Pow);

      case '.':
        return makeToken(TokenKind::Dot);

      case '<':
        return makeToken(TokenKind::Less);

      case '>':
        return makeToken(TokenKind::Greater);

      case ',':
        return makeToken(TokenKind::Comma);

      case ';':
        return makeToken(TokenKind::Semicolon);

      case ':':
        return makeToken(TokenKind::Colon);

      case '(':
        return makeToken(TokenKind::LPar);

      case ')':
        return makeToken(TokenKind::RPar);

      case '[':
        return makeToken(TokenKind::LSquare);

      case ']':
        return makeToken(TokenKind::RSquare);

      case '{':
        return makeToken(TokenKind::LCurly);

      case '}':
        return makeToken(TokenKind::RCurly);

      case '=':
        return makeToken(TokenKind::EqualityOperator);
    }

    error = "Unexpected character " + std::to_string(current);
    return makeToken(TokenKind::Error, getError());
  }

  std::optional<Token> ModelicaStateMachine::step(char c)
  {
    advance(c);

    switch (state) {
      case State::Normal:
        return scan<State::Normal>();

      case State::End:
        return makeToken(TokenKind::EndOfFile);

      case State::ParsingIdentifier:
        return scan<State::ParsingIdentifier>();

      case State::ParsingQIdentifier:
        return scan<State::ParsingQIdentifier>();

      case State::ParsingNumber:
        return scan<State::ParsingNumber>();

      case State::ParsingFloat:
        return scan<State::ParsingFloat>();

      case State::ParsingFloatExponentSign:
        return scan<State::ParsingFloatExponentSign>();

      case State::ParsingFloatExponent:
        return scan<State::ParsingFloatExponent>();

      case State::ParsingString:
        return scan<State::ParsingString>();

      case State::ParsingSingleLineComment:
        return scan<State::ParsingSingleLineComment>();

      case State::ParsingMultiLineComment:
        return scan<State::ParsingMultiLineComment>();

      case State::ParsingBackSlash:
        return scan<State::ParsingBackSlash>();

      case State::ParsingQIdentifierBackSlash:
        return scan<State::ParsingQIdentifierBackSlash>();

      case State::ParsingElementWiseSymbol:
        return scan<State::ParsingElementWiseSymbol>();

      case State::ParsingEqualSymbol:
        return scan<State::ParsingEqualSymbol>();

      case State::ParsingLessSymbol:
        return scan<State::ParsingLessSymbol>();

      case State::ParsingGreaterSymbol:
        return scan<State::ParsingGreaterSymbol>();

      case State::ParsingColonSymbol:
        return scan<State::ParsingColonSymbol>();

      case State::IgnoredCharacter:
        return scan<State::IgnoredCharacter>();
    }

    llvm_unreachable("Unknown lexer state");
    return makeToken(TokenKind::Error, getError());
  }

  void ModelicaStateMachine::advance(char c)
  {
    current = next;
    next = c;
    ++currentPosition.column;

    if (current == '\n') {
      currentPosition.column = 0;
      ++currentPosition.line;
    }
  }

  void ModelicaStateMachine::setTokenBeginPosition()
  {
    beginPosition = currentPosition;
  }

  void ModelicaStateMachine::setTokenEndPosition()
  {
    endPosition = currentPosition;
  }

  Token ModelicaStateMachine::makeToken(TokenKind kind)
  {
    return Token(kind, SourceRange(beginPosition, endPosition));
  }
}
