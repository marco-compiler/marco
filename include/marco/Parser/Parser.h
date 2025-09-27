#ifndef MARCO_PARSER_PARSER_H
#define MARCO_PARSER_PARSER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Parser/Location.h"
#include "marco/Parser/ModelicaStateMachine.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace marco::parser {
template <typename T>
class ValueWrapper {
public:
  ValueWrapper(SourceRange location, T value)
      : location(std::move(location)), value(std::move(value)) {}

  SourceRange getLocation() { return location; }

  T &operator*() { return value; }

  const T &operator*() const { return value; }

  T &getValue() { return value; }

  const T &getValue() const { return value; }

private:
  SourceRange location;
  T value;
};

// Deduction guide for ValueWrapper.
template <typename T>
ValueWrapper(SourceRange, T) -> ValueWrapper<T>;

// An optional parse result.
template <typename T>
using ParseResult = std::optional<T>;

// An optional parse result wrapped with its location.
// Useful to provide a unique location for a set of multiple objects.
template <typename T>
using WrappedParseResult = std::optional<ValueWrapper<T>>;

/// The parser encapsulates the lexer but not the memory where the string we
/// are reading is held. It exposes parts of the grammatical rules that are
/// available in the grammar.
class Parser {
public:
  Parser(clang::DiagnosticsEngine &diagnosticsEngine,
         clang::SourceManager &sourceManager,
         std::shared_ptr<SourceFile> source);

  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseRoot();

  /// Parse a boolean value.
  WrappedParseResult<bool> parseBoolValue();

  /// Parse an integer value.
  WrappedParseResult<int64_t> parseIntValue();

  /// Parse a floating point value.
  WrappedParseResult<double> parseFloatValue();

  /// Parse a string.
  WrappedParseResult<std::string> parseString();

  /// Parse the name of an identifier.
  WrappedParseResult<std::string> parseIdentifier();

  /// Parse the 'class-definition' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseClassDefinition();

  /// Parse the 'external-function-call' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>>
  parseExternalFunctionCall();

  /// Parse the 'modification' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseModification();

  /// Parse the 'class-modification' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>>
  parseClassModification();

  /// Parse the 'argument-list' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::bmodelica::ASTNode>>>
  parseArgumentList();

  /// Parse the 'argument' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseArgument();

  /// Parse the 'element-modification' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>>
  parseElementModification(bool each, bool final);

  /// Parse the 'element-redeclaration' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>>
  parseElementRedeclaration();

  /// Parse the 'element-replaceable' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>>
  parseElementReplaceable(bool each, bool final);

  /// Parse the 'algorithm-section' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseAlgorithmSection();

  /// Parse the 'equation-section' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseEquationSection();

  /// Parse the 'equation' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseEquation();

  /// Parse the 'statement' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseStatement();

  /// Parse the 'if-equation' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseIfEquation();

  /// Parse the 'if-statement' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseIfStatement();

  /// Parse the 'for-equation' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseForEquation();

  /// Parse the 'for-statement' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseForStatement();

  /// Parse the 'for-indices' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::bmodelica::ASTNode>>>
  parseForIndices();

  /// Parse the 'for-index' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseForIndex();

  /// Parse the 'while-statement' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseWhileStatement();

  /// Parse the 'when-equation' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseWhenEquation();

  /// Parse the 'when-statement' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseWhenStatement();

  /// Parse the 'expression' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseExpression();

  /// Parse the 'simple-expression' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseSimpleExpression();

  /// Parse the 'logical-expression' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>>
  parseLogicalExpression();

  /// Parse the 'logical-term' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseLogicalTerm();

  /// Parse the 'logical-factor' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseLogicalFactor();

  /// Parse the 'relation' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseRelation();

  /// Parse the 'relational-operator' symbol.
  WrappedParseResult<ast::bmodelica::OperationKind> parseRelationalOperator();

  /// Parse the 'arithmetic-expression' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>>
  parseArithmeticExpression();

  /// Parse the 'add-operator' symbol.
  WrappedParseResult<ast::bmodelica::OperationKind> parseAddOperator();

  /// Parse the 'term' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseTerm();

  /// Parse the 'mul-operator' symbol.
  WrappedParseResult<ast::bmodelica::OperationKind> parseMulOperator();

  /// Parse the 'factor' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseFactor();

  /// Parse the 'primary' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parsePrimary();

  /// Parse the 'component-reference' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>>
  parseComponentReference();

  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>>
  parseComponentReferenceEntry();

  /// Parse the 'function-call-args' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::bmodelica::ASTNode>>>
  parseFunctionCallArgs();

  /// Parse the 'function-arguments' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::bmodelica::ASTNode>>>
  parseFunctionArguments();

  /// Parse the 'function-arguments-non-first' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::bmodelica::ASTNode>>>
  parseFunctionArgumentsNonFirst();

  /// Parse the 'array-arguments' symbol.
  ParseResult<std::pair<std::vector<std::unique_ptr<ast::bmodelica::ASTNode>>,
                        std::vector<std::unique_ptr<ast::bmodelica::ASTNode>>>>
  parseArrayArguments();

  /// Parse the 'array-arguments-non-first' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::bmodelica::ASTNode>>>
  parseArrayArgumentsNonFirst();

  /// Parse the 'named-arguments' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::bmodelica::ASTNode>>>
  parseNamedArguments();

  /// Parse the 'named-argument' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseNamedArgument();

  /// Parse the 'function-argument' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseFunctionArgument();

  /// Parse the 'output-expression-list' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::bmodelica::ASTNode>>>
  parseOutputExpressionList();

  /// Parse the 'array-subscripts' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::bmodelica::ASTNode>>>
  parseArraySubscripts();

  /// Parse the 'subscript' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseSubscript();

  /// Parse the 'annotation' symbol.
  ParseResult<std::unique_ptr<ast::bmodelica::ASTNode>> parseAnnotation();

private:
  std::optional<std::vector<std::unique_ptr<ast::bmodelica::ASTNode>>>
  parseElementList(bool publicSection);

  std::optional<std::unique_ptr<ast::bmodelica::ASTNode>>
  parseElement(bool publicSection);

  std::optional<std::unique_ptr<ast::bmodelica::ASTNode>> parseTypePrefix();

  std::optional<std::unique_ptr<ast::bmodelica::ASTNode>> parseVariableType();

  std::optional<std::unique_ptr<ast::bmodelica::ASTNode>> parseArrayDimension();

  std::optional<std::unique_ptr<ast::bmodelica::ASTNode>>
  parseTermModification();

private:
  /// Move to the next token.
  void advance();

  /// Check if the first lookahead has the requested type.
  /// If it does, shift the tokens. Otherwise, keep the current state.
  template <TokenKind Kind>
  bool accept() {
    if (lookahead.front().getKind() == Kind) {
      advance();
      return true;
    }

    return false;
  }

  /// Get the source location of the current token.
  [[nodiscard]] SourceRange getLocation() const;

  /// Get the current parsing location.
  [[nodiscard]] SourceRange getCursorLocation() const;

  /// Get the string value stored with the current token.
  [[nodiscard]] std::string getString() const;

  /// Get the integer value stored with the current token.
  [[nodiscard]] int64_t getInt() const;

  /// Get the floating point value stored with the current token.
  [[nodiscard]] double getFloat() const;

  /// @name Diagnostics
  /// {

  clang::SourceLocation convertLocation(const SourceRange &location) const;

  void emitUnexpectedTokenError(const Token &found, TokenKind expected);

  void emitUnexpectedIdentifierError(const SourceRange &location,
                                     llvm::StringRef found,
                                     llvm::StringRef expected);

  /// }

private:
  clang::DiagnosticsEngine *diagnosticsEngine;
  clang::SourceManager *sourceManager;
  Lexer<ModelicaStateMachine> lexer;
  Token token{TokenKind::Begin};
  llvm::SmallVector<Token, 2> lookahead{2, Token(TokenKind::Begin)};
};
} // namespace marco::parser

#endif // MARCO_PARSER_PARSER_H
