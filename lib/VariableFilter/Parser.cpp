#include "marco/VariableFilter/Parser.h"
#include "marco/VariableFilter/Error.h"
#include "llvm/Support/Regex.h"

using namespace ::marco;
using namespace ::marco::vf;

#define EXPECT(Token)                                                                           \
	if (!accept<Token>()) {                                                                       \
	  diagnostics->emitError<UnexpectedTokenMessage>(lexer.getTokenPosition(), current, Token);   \
    return std::nullopt;                                                                        \
  }                                                                                             \
  static_assert(true)

#define TRY(outVar, expression)       \
	auto outVar = expression;           \
	if (!outVar.has_value()) {          \
    return std::nullopt;              \
  }                                   \
  static_assert(true)

namespace marco::vf
{
  Parser::Parser(VariableFilter& vf, diagnostic::DiagnosticEngine* diagnostics, std::shared_ptr<SourceFile> file)
      : vf(&vf),
        diagnostics(diagnostics),
        lexer(file),
        current(lexer.scan())
  {
  }

  bool Parser::run()
  {
    // In case of empty string
    if (current == Token::EndOfFile) {
      return true;
    }

    // Consume consecutive semicolons
    while(accept<Token::Semicolons>());

    // Check if we reached the end of the string
    if (current == Token::EndOfFile) {
      return true;
    }

    if (!token()) {
      return false;
    }

    while (accept<Token::Semicolons>()) {
      // For strings ending with a semicolon
      if (current == Token::EndOfFile) {
        return true;
      }

      if (current != Token::Semicolons) {
        if (!token()) {
          return false;
        }
      }
    }

    return true;
  }

  bool Parser::token()
  {
    if (current == Token::DerKeyword) {
      auto derNode = der();

      if (!derNode.has_value()) {
        return false;
      }

      Tracker tracker(derNode->getDerivedVariable().getIdentifier());
      vf->addDerivative(tracker);
      return true;

    } else if (current == Token::Regex) {
      auto regexNode = regex();

      if (!regexNode.has_value()) {
        return false;
      }

      vf->addRegexString(regexNode->getRegex());
      return true;
    }

    auto variableNode = identifier();

    if (!variableNode.has_value()) {
      return false;
    }

    if (current == Token::LSquare) {
      auto arrayNode = array(*variableNode);

      if (!arrayNode.has_value()) {
        return false;
      }

      Tracker tracker(arrayNode->getVariable().getIdentifier(), arrayNode->getRanges());
      vf->addVariable(tracker);
      return true;
    }

    vf->addVariable(Tracker(variableNode->getIdentifier()));
    return true;
  }

  std::optional<DerivativeExpression> Parser::der()
  {
    EXPECT(Token::DerKeyword);
    EXPECT(Token::LPar);
    VariableExpression variable(lexer.getLastIdentifier());
    EXPECT(Token::Ident);
    EXPECT(Token::RPar);
    return DerivativeExpression(variable);
  }

  std::optional<RegexExpression> Parser::regex()
  {
    auto loc = lexer.getTokenPosition();
    RegexExpression node(lexer.getLastRegex());
    EXPECT(Token::Regex);

    if (node.getRegex().empty()) {
      diagnostics->emitError<EmptyRegexMessage>(loc);
      return std::nullopt;
    }

    llvm::Regex regexObj(node.getRegex());

    if (!regexObj.isValid()) {
      diagnostics->emitError<InvalidRegexMessage>(loc);
      return std::nullopt;
    }

    return node;
  }

  std::optional<VariableExpression> Parser::identifier()
  {
    VariableExpression node(lexer.getLastIdentifier());
    EXPECT(Token::Ident);
    return node;
  }

  std::optional<ArrayExpression> Parser::array(VariableExpression variable)
  {
    EXPECT(Token::LSquare);

    llvm::SmallVector<Range, 3> ranges;
    TRY(range, arrayRange());
    ranges.push_back(*range);

    while (accept<Token::Comma>()) {
      TRY(anotherRange, arrayRange());
      ranges.push_back(*anotherRange);
    }

    EXPECT(Token::RSquare);
    return ArrayExpression(variable, ranges);
  }

  std::optional<Range> Parser::arrayRange()
  {
    auto getIndex = [&]() -> std::optional<int> {
      if (accept<Token::Dollar>()) {
        return Range::kUnbounded;
      }

      int index = lexer.getLastInt();
      EXPECT(Token::Integer);
      return index;
    };

    TRY(lowerBound, getIndex());
    EXPECT(Token::Colons);
    TRY(upperBound, getIndex());

    return Range(*lowerBound, *upperBound);
  }

  void Parser::next()
  {
    current = lexer.scan();
  }

  bool Parser::accept(Token t)
  {
    if (current == t)
    {
      next();
      return true;
    }

    return false;
  }
}
