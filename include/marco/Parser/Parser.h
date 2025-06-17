#ifndef MARCO_PARSER_PARSER_H
#define MARCO_PARSER_PARSER_H

#include "marco/AST/AST.h"
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

  ParseResult<std::unique_ptr<ast::ASTNode>> parseRoot();

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
  ParseResult<std::unique_ptr<ast::ASTNode>> parseClassDefinition();

  /// Parse the 'modification' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseModification();

  /// Parse the 'class-modification' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseClassModification();

  /// Parse the 'argument-list' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::ASTNode>>>
  parseArgumentList();

  /// Parse the 'argument' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseArgument();

  /// Parse the 'element-modification' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>>
  parseElementModification(bool each, bool final);

  /// Parse the 'element-redeclaration' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseElementRedeclaration();

  /// Parse the 'element-replaceable' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>>
  parseElementReplaceable(bool each, bool final);

  /// Parse the 'algorithm-section' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseAlgorithmSection();

  /// Parse the 'equation-section' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseEquationSection();

  /// Parse the 'equation' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseEquation();

  /// Parse the 'statement' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseStatement();

  /// Parse the 'if-equation' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseIfEquation();

  /// Parse the 'if-statement' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseIfStatement();

  /// Parse the 'for-equation' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseForEquation();

  /// Parse the 'for-statement' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseForStatement();

  /// Parse the 'for-indices' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::ASTNode>>>
  parseForIndices();

  /// Parse the 'for-index' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseForIndex();

  /// Parse the 'while-statement' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseWhileStatement();

  /// Parse the 'when-equation' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseWhenEquation();

  /// Parse the 'when-statement' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseWhenStatement();

  /// Parse the 'expression' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseExpression();

  /// Parse the 'simple-expression' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseSimpleExpression();

  /// Parse the 'logical-expression' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseLogicalExpression();

  /// Parse the 'logical-term' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseLogicalTerm();

  /// Parse the 'logical-factor' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseLogicalFactor();

  /// Parse the 'relation' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseRelation();

  /// Parse the 'relational-operator' symbol.
  WrappedParseResult<ast::OperationKind> parseRelationalOperator();

  /// Parse the 'arithmetic-expression' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseArithmeticExpression();

  /// Parse the 'add-operator' symbol.
  WrappedParseResult<ast::OperationKind> parseAddOperator();

  /// Parse the 'term' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseTerm();

  /// Parse the 'mul-operator' symbol.
  WrappedParseResult<ast::OperationKind> parseMulOperator();

  /// Parse the 'factor' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseFactor();

  /// Parse the 'primary' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parsePrimary();

  /// Parse the 'component-reference' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseComponentReference();

  ParseResult<std::unique_ptr<ast::ASTNode>> parseComponentReferenceEntry();

  /// Parse the 'function-call-args' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::ASTNode>>>
  parseFunctionCallArgs();

  /// Parse the 'function-arguments' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::ASTNode>>>
  parseFunctionArguments();

  /// Parse the 'function-arguments-non-first' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::ASTNode>>>
  parseFunctionArgumentsNonFirst();

  /// Parse the 'array-arguments' symbol.
  ParseResult<std::pair<std::vector<std::unique_ptr<ast::ASTNode>>,
                        std::vector<std::unique_ptr<ast::ASTNode>>>>
  parseArrayArguments();

  /// Parse the 'array-arguments-non-first' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::ASTNode>>>
  parseArrayArgumentsNonFirst();

  /// Parse the 'named-arguments' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::ASTNode>>>
  parseNamedArguments();

  /// Parse the 'named-argument' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseNamedArgument();

  /// Parse the 'function-argument' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseFunctionArgument();

  /// Parse the 'output-expression-list' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::ASTNode>>>
  parseOutputExpressionList();

  /// Parse the 'array-subscripts' symbol.
  WrappedParseResult<std::vector<std::unique_ptr<ast::ASTNode>>>
  parseArraySubscripts();

  /// Parse the 'subscript' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseSubscript();

  /// Parse the 'annotation' symbol.
  ParseResult<std::unique_ptr<ast::ASTNode>> parseAnnotation();

private:
  std::optional<std::vector<std::unique_ptr<ast::ASTNode>>>
  parseElementList(bool publicSection);

  std::optional<std::unique_ptr<ast::ASTNode>> parseElement(bool publicSection);

  std::optional<std::unique_ptr<ast::ASTNode>> parseTypePrefix();

  std::optional<std::unique_ptr<ast::ASTNode>> parseVariableType();

  std::optional<std::unique_ptr<ast::ASTNode>> parseArrayDimension();

  std::optional<std::unique_ptr<ast::ASTNode>> parseTermModification();

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
