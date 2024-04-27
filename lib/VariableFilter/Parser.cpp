#include "marco/VariableFilter/Parser.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Regex.h"

using namespace ::marco::vf;

#define EXPECT(Token)                               \
	if (!accept<Token>()) {                           \
    emitUnexpectedTokenError(lookahead[0], Token);  \
    return std::nullopt;                            \
  }                                                 \
  static_assert(true)

#define TRY(outVar, expression)       \
	auto outVar = expression;           \
	if (!outVar.has_value()) {          \
    return std::nullopt;              \
  }                                   \
  static_assert(true)

namespace marco::vf
{
  Parser::Parser(
      VariableFilter& vf,
      clang::DiagnosticsEngine& diagnosticsEngine,
      clang::SourceManager& sourceManager,
      std::shared_ptr<SourceFile> source)
      : vf(&vf),
        diagnosticsEngine(&diagnosticsEngine),
        sourceManager(&sourceManager),
        lexer(source)
  {
    for (size_t i = 0, e = lookahead.size(); i < e; ++i) {
      advance();
    }
  }

  void Parser::advance()
  {
    token = lookahead[0];

    for (size_t i = 0, e = lookahead.size(); i + 1 < e; ++i) {
      lookahead[i] = lookahead[i + 1];
    }

    lookahead.back() = lexer.scan();
  }

  SourceRange Parser::getLocation() const
  {
    return token.getLocation();
  }

  SourceRange Parser::getCursorLocation() const
  {
    return {getLocation().end, getLocation().end};
  }

  std::string Parser::getString() const
  {
    return token.getString();
  }

  int64_t Parser::getInt() const
  {
    return token.getInt();
  }

  clang::SourceLocation Parser::convertLocation(
      const SourceRange& location) const
  {
    auto& fileManager = sourceManager->getFileManager();

    auto fileRef = fileManager.getVirtualFileRef(
        location.begin.file->getFileName(), 0, 0);

    return sourceManager->translateFileLineCol(
        fileRef, location.begin.line, location.begin.column);
  }

  void Parser::emitUnexpectedTokenError(const Token& found, TokenKind expected)
  {
    auto& diags = *diagnosticsEngine;
    auto location = convertLocation(found.getLocation());

    auto diagID = diags.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "Unexpected token: found '%0' instead of '%1'");

    diags.Report(location, diagID) << toString(found) << toString(expected);
  }

  void Parser::emitEmptyRegexError(const SourceRange& location)
  {
    auto& diags = *diagnosticsEngine;
    auto convertedLocation = convertLocation(location);

    auto diagID = diags.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "Empty regular expression");

    diags.Report(convertedLocation, diagID);
  }

  void Parser::emitInvalidRegexError(const SourceRange& location)
  {
    auto& diags = *diagnosticsEngine;
    auto convertedLocation = convertLocation(location);

    auto diagID = diags.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "Invalid regular expression");

    diags.Report(convertedLocation, diagID);
  }

  bool Parser::run()
  {
    while (!lookahead[0].isa<TokenKind::EndOfFile>()) {
      // Consume consecutive semicolons.
      if (accept<TokenKind::Semicolons>()) {
        continue;
      }

      if (!parseListElement()) {
        return false;
      }
    }

    return true;
  }

  bool Parser::parseListElement()
  {
    if (lookahead[0].isa<TokenKind::DerKeyword>()) {
      auto derNode = parseDer();

      if (!derNode.has_value()) {
        return false;
      }

      Tracker tracker(derNode->getDerivedVariable().getIdentifier());
      vf->addDerivative(tracker);
      return true;
    }

    if (lookahead[0].isa<TokenKind::Regex>()) {
      auto regexNode = parseRegex();

      if (!regexNode.has_value()) {
        return false;
      }

      vf->addRegexString(regexNode->getRegex());
      return true;
    }

    auto variableNode = parseVariableExpression();

    if (!variableNode.has_value()) {
      return false;
    }

    if (lookahead[0].isa<TokenKind::LSquare>()) {
      auto arrayNode = parseArray(*variableNode);

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

  std::optional<RegexExpression> Parser::parseRegex()
  {
    EXPECT(TokenKind::Regex);
    SourceRange loc = getLocation();
    RegexExpression node(getString());

    if (node.getRegex().empty()) {
      emitEmptyRegexError(loc);
      return std::nullopt;
    }

    llvm::Regex regexObj(node.getRegex());

    if (!regexObj.isValid()) {
      emitInvalidRegexError(loc);
      return std::nullopt;
    }

    return node;
  }

  std::optional<VariableExpression> Parser::parseVariableExpression()
  {
    EXPECT(TokenKind::Identifier);
    VariableExpression node(getString());
    return node;
  }

  std::optional<ArrayExpression> Parser::parseArray(
      VariableExpression variable)
  {
    EXPECT(TokenKind::LSquare);

    llvm::SmallVector<Range, 3> ranges;
    TRY(range, parseArrayRange());
    ranges.push_back(*range);

    while (accept<TokenKind::Comma>()) {
      TRY(anotherRange, parseArrayRange());
      ranges.push_back(*anotherRange);
    }

    EXPECT(TokenKind::RSquare);
    return ArrayExpression(std::move(variable), ranges);
  }

  std::optional<Range> Parser::parseArrayRange()
  {
    TRY(lowerBound, parseArrayIndex());
    EXPECT(TokenKind::Colons);
    TRY(upperBound, parseArrayIndex());
    return Range(*lowerBound, *upperBound);
  }

  std::optional<int> Parser::parseArrayIndex()
  {
    if (accept<TokenKind::Dollar>()) {
      return Range::kUnbounded;
    }

    EXPECT(TokenKind::Integer);
    return getInt();
  }

  std::optional<DerivativeExpression> Parser::parseDer()
  {
    EXPECT(TokenKind::DerKeyword);
    EXPECT(TokenKind::LPar);
    EXPECT(TokenKind::Identifier);
    VariableExpression variable(getString());
    EXPECT(TokenKind::RPar);
    return DerivativeExpression(std::move(variable));
  }
}
