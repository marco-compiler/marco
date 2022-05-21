#include "marco/VariableFilter/Parser.h"
#include "marco/VariableFilter/Error.h"
#include "llvm/Support/Regex.h"

using namespace ::marco;
using namespace ::marco::vf;

#define EXPECT(Token)							\
	if (auto e = expect(Token); !e)	\
	return e.takeError()

#define TRY(outVar, expression)		\
	auto outVar = expression;				\
	if (!outVar)										\
	return outVar.takeError()

namespace marco::vf
{
  Parser::Parser(VariableFilter& vf, llvm::StringRef source)
      : vf(&vf),
        lexer(source.data()),
        current(lexer.scan()),
        tokenRange("-", source.data(), 1, 1, 1, 1)
  {
    updateTokenSourceRange();
  }

  llvm::Error Parser::run()
  {
    // In case of empty string
    if (current == Token::EndOfFile) {
      return llvm::Error::success();
    }

    // Consume consecutive semicolons
    while(accept<Token::Semicolons>());

    // Check if we reached the end of the string
    if (current == Token::EndOfFile) {
      return llvm::Error::success();
    }

    if (auto error = token()) {
      return error;
    }

    while (accept<Token::Semicolons>()) {
      // For strings ending with a semicolon
      if (current == Token::EndOfFile) {
        return llvm::Error::success();
      }

      if (current != Token::Semicolons) {
        if (auto error = token()) {
          return error;
        }
      }
    }

    return llvm::Error::success();
  }

  llvm::Error Parser::token()
  {
    if (current == Token::DerKeyword) {
      TRY(derNode, der());
      Tracker tracker(derNode->getDerivedVariable().getIdentifier());
      vf->addDerivative(tracker);
      return llvm::Error::success();
    } else if (current == Token::Regex) {
      TRY(regexNode, regex());
      vf->addRegexString(regexNode->getRegex());
      return llvm::Error::success();
    }

    TRY(variableNode, identifier());

    if (current == Token::LSquare) {
      TRY(arrayNode, array(*variableNode));
      Tracker tracker(arrayNode->getVariable().getIdentifier(), arrayNode->getRanges());
      vf->addVariable(tracker);
      return llvm::Error::success();
    }

    vf->addVariable(Tracker(variableNode->getIdentifier()));
    return llvm::Error::success();
  }

  llvm::Expected<DerivativeExpression> Parser::der()
  {
    EXPECT(Token::DerKeyword);
    EXPECT(Token::LPar);
    VariableExpression variable(lexer.getLastIdentifier());
    EXPECT(Token::Ident);
    EXPECT(Token::RPar);
    return DerivativeExpression(variable);
  }

  llvm::Expected<RegexExpression> Parser::regex()
  {
    RegexExpression node(lexer.getLastRegex());
    EXPECT(Token::Regex);

    if (node.getRegex().empty()) {
      return llvm::make_error<EmptyRegex>(tokenRange);
    }

    llvm::Regex regexObj(node.getRegex());

    if (!regexObj.isValid()) {
      return llvm::make_error<InvalidRegex>(tokenRange);
    }

    return node;
  }

  llvm::Expected<VariableExpression> Parser::identifier()
  {
    VariableExpression node(lexer.getLastIdentifier());
    EXPECT(Token::Ident);
    return node;
  }

  llvm::Expected<ArrayExpression> Parser::array(VariableExpression variable)
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

  llvm::Expected<Range> Parser::arrayRange()
  {
    auto getIndex = [&]() -> llvm::Expected<int> {
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
    updateTokenSourceRange();
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

  llvm::Expected<bool> Parser::expect(Token t)
  {
    if (accept(t)) {
      return true;
    }

    return llvm::make_error<UnexpectedToken>(tokenRange, current);
  }

  void Parser::updateTokenSourceRange()
  {
    tokenRange.startLine = lexer.getTokenStartLine();
    tokenRange.startColumn = lexer.getTokenStartColumn();
    tokenRange.endLine = lexer.getTokenEndLine();
    tokenRange.endColumn = lexer.getTokenEndColumn();
  }
}
