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
    reservedKeywords["algorithm"] = Token::Algorithm;
    reservedKeywords["and"] = Token::And;
    reservedKeywords["annotation"] = Token::Annotation;
    reservedKeywords["block"] = Token::Block;
    reservedKeywords["break"] = Token::Break;
    reservedKeywords["class"] = Token::Class;
    reservedKeywords["connect"] = Token::Connect;
    reservedKeywords["connector"] = Token::Connector;
    reservedKeywords["constant"] = Token::Constant;
    reservedKeywords["constrainedby"] = Token::ConstrainedBy;
    reservedKeywords["der"] = Token::Der;
    reservedKeywords["discrete"] = Token::Discrete;
    reservedKeywords["each"] = Token::Each;
    reservedKeywords["else"] = Token::Else;
    reservedKeywords["elseif"] = Token::ElseIf;
    reservedKeywords["elsewhen"] = Token::ElseWhen;
    reservedKeywords["encapsulated"] = Token::Encapsulated;
    reservedKeywords["end"] = Token::End;
    reservedKeywords["enumeration"] = Token::Enumeration;
    reservedKeywords["equation"] = Token::Equation;
    reservedKeywords["expandable"] = Token::Expandable;
    reservedKeywords["extends"] = Token::Extends;
    reservedKeywords["external"] = Token::External;
    reservedKeywords["false"] = Token::False;
    reservedKeywords["final"] = Token::Final;
    reservedKeywords["flow"] = Token::Flow;
    reservedKeywords["for"] = Token::For;
    reservedKeywords["function"] = Token::Function;
    reservedKeywords["if"] = Token::If;
    reservedKeywords["import"] = Token::Import;
    reservedKeywords["impure"] = Token::Impure;
    reservedKeywords["in"] = Token::In;
    reservedKeywords["initial"] = Token::Initial;
    reservedKeywords["inner"] = Token::Inner;
    reservedKeywords["input"] = Token::Input;
    reservedKeywords["loop"] = Token::Loop;
    reservedKeywords["model"] = Token::Model;
    reservedKeywords["not"] = Token::Not;
    reservedKeywords["operator"] = Token::Operator;
    reservedKeywords["or"] = Token::Or;
    reservedKeywords["outer"] = Token::Outer;
    reservedKeywords["output"] = Token::Output;
    reservedKeywords["package"] = Token::Package;
    reservedKeywords["parameter"] = Token::Parameter;
    reservedKeywords["partial"] = Token::Partial;
    reservedKeywords["protected"] = Token::Protected;
    reservedKeywords["public"] = Token::Public;
    reservedKeywords["pure"] = Token::Pure;
    reservedKeywords["record"] = Token::Record;
    reservedKeywords["redeclare"] = Token::Redeclare;
    reservedKeywords["replaceable"] = Token::Replaceable;
    reservedKeywords["return"] = Token::Return;
    reservedKeywords["stream"] = Token::Stream;
    reservedKeywords["then"] = Token::Then;
    reservedKeywords["true"] = Token::True;
    reservedKeywords["type"] = Token::Type;
    reservedKeywords["when"] = Token::When;
    reservedKeywords["while"] = Token::While;
    reservedKeywords["within"] = Token::Within;
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

  std::string ModelicaStateMachine::getError() const
  {
    return error;
  }

  SourcePosition ModelicaStateMachine::getCurrentPosition() const
  {
    return currentPosition;
  }

  SourceRange ModelicaStateMachine::getTokenPosition() const
  {
    return SourceRange(beginPosition, endPosition);
  }

  Token ModelicaStateMachine::getNoneToken() const
  {
    return Token::None;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingIdentifier>()
  {
    setTokenEndPosition();
    identifier.push_back(current);

    if (!isDigit(next) && !isNonDigit(next)) {
      state = State::Normal;

      if (auto it = reservedKeywords.find(identifier); it != reservedKeywords.end()) {
        return it->second;
      }

      return Token::Identifier;
    }

    return Token::None;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingQIdentifier>()
  {
    setTokenEndPosition();

    if (current == '\\') {
      state = State::ParsingQIdentifierBackSlash;
      return Token::None;
    }

    if (current == '\'') {
      state = State::Normal;
      return Token::Identifier;
    }

    if (next == '\0') {
      state = State::Normal;
      error = "unexpected end of file while parsing a q-identifier";
      return Token::Error;
    }

    identifier.push_back(current);
    return Token::None;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingNumber>()
  {
    setTokenEndPosition();

    if (isDigit(current)) {
      numberLexer.addUpper(current - '0');
    }

    if (current == '.') {
      state = State::ParsingFloat;
      return Token::None;
    }

    if (current == 'E' || current == 'e') {
      if (next == '+' || next == '-') {
        state = State::ParsingFloatExponentSign;
        return Token::None;
      }

      if (isDigit(next)) {
        numberLexer.setSign(true);
        state = State::ParsingFloatExponent;
        return Token::None;
      }

      state = State::Normal;
      error = "missing exponent in floating point number";
      return Token::Error;
    }

    auto isAcceptable = [](char c) {
      return isDigit(c) || c == '.' || c == 'e' || c == 'E';
    };

    if (!isAcceptable(next)) {
      state = State::Normal;
      return Token::Integer;
    }

    return Token::None;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingFloat>()
  {
    if (std::isspace(current) || current == '\0') {
      state = State::Normal;
      return Token::FloatingPoint;
    }

    if (isDigit(current)) {
      numberLexer.addLower(current - '0');
    }

    if (current == 'E' || current == 'e') {
      if (next == '+' || next == '-') {
        state = State::ParsingFloatExponentSign;
        return Token::None;
      }

      if (isDigit(next)) {
        numberLexer.setSign(true);
        state = State::ParsingFloatExponent;
        return Token::None;
      }
    }

    if (isDigit(next) || next == 'E' || next == 'e') {
      return Token::None;
    }

    state = State::Normal;
    setTokenEndPosition();
    return Token::FloatingPoint;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingFloatExponentSign>()
  {
    if (std::isspace(current) || current == '\0') {
      state = State::Normal;
      error = "unexpected termination of the floating point number";
      return Token::Error;
    }

    setTokenEndPosition();

    if (current == '-' || current == '+') {
      if (!isDigit(next)) {
        error = "Exponent sign must be followed by a number";
        state = State::Normal;
        return Token::Error;
      }

      state = State::ParsingFloatExponent;
      numberLexer.setSign(current == '+');
      return Token::None;
    }

    error = "unexpected character '" + std::to_string(current) + "' in floating pointer number scan";
    state = State::Normal;
    return Token::Error;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingFloatExponent>()
  {
    if (std::isspace(current) || current == '\0') {
      state = State::Normal;
      //error = "unexpected termination of the floating point number";
      return Token::FloatingPoint;
    }

    setTokenEndPosition();

    if (isDigit(current)) {
      numberLexer.addExponential(current - '0');
    }

    if (isDigit(next)) {
      return Token::None;
    }

    state = State::Normal;
    return Token::FloatingPoint;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingString>()
  {
    if (current == '\0') {
      state = State::End;
      error = "reached end of file while parsing a string";
      return Token::Error;
    }

    setTokenEndPosition();

    if (current == '"') {
      state = State::Normal;
      return Token::String;
    }

    if (current == '\\') {
      state = State::ParsingBackSlash;
      return Token::None;
    }

    stringValue.push_back(current);
    return Token::None;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingSingleLineComment>()
  {
    setTokenEndPosition();

    if (next == '\0') {
      state = State::Normal;
    }

    if (current == '\n') {
      state = State::Normal;
    }

    return Token::None;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingMultiLineComment>()
  {
    setTokenEndPosition();

    if (next == '\0') {
      state = State::Normal;
    }

    if (current == '*' && next == '/') {
      state = State::IgnoredCharacter;
    }

    return Token::None;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingBackSlash>()
  {
    setTokenEndPosition();
    stringValue.push_back(escapedChar(current));
    state = State::ParsingString;
    return Token::None;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingQIdentifierBackSlash>()
  {
    setTokenEndPosition();
    identifier.push_back(escapedChar(current));
    state = State::ParsingQIdentifier;
    return Token::None;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingElementWiseSymbol>()
  {
    setTokenEndPosition();
    state = State::Normal;

    if (current == '+') {
      return Token::PlusEW;
    }

    if (current == '-') {
      return Token::MinusEW;
    }

    if (current == '*') {
      return Token::ProductEW;
    }

    if (current == '/') {
      return Token::DivisionEW;
    }

    if (current == '^') {
      return Token::PowEW;
    }

    error = "unexpected character '" + std::to_string(current) + "'";
    return Token::Error;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingEqualSymbol>()
  {
    setTokenEndPosition();
    state = State::Normal;

    if (current == '=') {
      return Token::Equal;
    }

    error = "unexpected character '" + std::to_string(current) + "'";
    return Token::Error;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingLessSymbol>()
  {
    setTokenEndPosition();
    state = State::Normal;

    if (current == '>') {
      return Token::NotEqual;
    }

    if (current == '=') {
      return Token::LessEqual;
    }

    error = "unexpected character '" + std::to_string(current) + "'";
    return Token::Error;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingGreaterSymbol>()
  {
    setTokenEndPosition();
    state = State::Normal;

    if (current == '=') {
      return Token::GreaterEqual;
    }

    error = "unexpected character '" + std::to_string(current) + "'";
    return Token::Error;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::ParsingColonSymbol>()
  {
    setTokenEndPosition();
    state = State::Normal;

    if (current == '=') {
      return Token::AssignmentOperator;
    }

    error = "unexpected character '" + std::to_string(current) + "'";
    return Token::Error;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::IgnoredCharacter>()
  {
    setTokenEndPosition();
    state = State::Normal;
    return Token::None;
  }

  template<>
  Token ModelicaStateMachine::scan<ModelicaStateMachine::State::Normal>()
  {
    if (std::isspace(current) != 0) {
      // Skip spaces
      return Token::None;
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
      return Token::None;
    }

    if (isDigit(current)) {
      state = State::ParsingNumber;
      numberLexer.reset();
      return scan<State::ParsingNumber>();
    }

    if (current == '"') {
      state = State::ParsingString;
      stringValue = "";
      return Token::None;
    }

    if (current == '/' && next == '/') {
      state = State::ParsingSingleLineComment;
      return Token::None;
    }

    if (current == '/' && next == '*') {
      state = State::ParsingMultiLineComment;
      return Token::None;
    }

    if (current == '\0') {
      state = State::End;
      return Token::EndOfFile;
    }

    return trySymbolScan();
  }

  Token ModelicaStateMachine::trySymbolScan()
  {
    if (current == '.') {
      switch (next) {
        case '+':
        case '-':
        case '*':
        case '/':
        case '^':
          state = State::ParsingElementWiseSymbol;
          return Token::None;
      }
    }

    if (current == '=' && next == '=') {
      state = State::ParsingEqualSymbol;
      return Token::None;
    }

    if (current == '<') {
      switch (next) {
        case '>':
        case '=':
          state = State::ParsingLessSymbol;
          return Token::None;
      }
    }

    if (current == '>' && next == '=') {
      state = State::ParsingGreaterSymbol;
      return Token::None;
    }

    if (current == ':' && next == '=') {
      state = State::ParsingColonSymbol;
      return Token::None;
    }

    state = State::Normal;

    switch (current) {
      case '+':
        return Token::Plus;

      case '-':
        return Token::Minus;

      case '*':
        return Token::Product;

      case '/':
        return Token::Division;

      case '^':
        return Token::Pow;

      case '.':
        return Token::Dot;

      case '<':
        return Token::Less;

      case '>':
        return Token::Greater;

      case ',':
        return Token::Comma;

      case ';':
        return Token::Semicolon;

      case ':':
        return Token::Colon;

      case '(':
        return Token::LPar;

      case ')':
        return Token::RPar;

      case '[':
        return Token::LSquare;

      case ']':
        return Token::RSquare;

      case '{':
        return Token::LCurly;

      case '}':
        return Token::RCurly;

      case '=':
        return Token::EqualityOperator;
    }

    error = "Unexpected character " + std::to_string(current);
    return Token::Error;
  }

  Token ModelicaStateMachine::step(char c)
  {
    advance(c);

    switch (state) {
      case State::Normal:
        return scan<State::Normal>();

      case State::End:
        return Token::EndOfFile;

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
    return Token::Error;
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
}
