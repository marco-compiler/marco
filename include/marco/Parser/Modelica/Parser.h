#ifndef MARCO_PARSER_MODELICA_PARSER_H
#define MARCO_PARSER_MODELICA_PARSER_H

#include "marco/AST/Modelica/AST.h"
#include "marco/Parser/Modelica/Token.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include <memory>
#include <vector>

namespace marco::parser::modelica {
/// Recursive descent parser for the Modelica language, following the grammar
/// of the language specification. Parsing stops at the first error, which is reported
/// through the diagnostics engine. The parse methods return a null pointer or
/// false in case of error.
class Parser {
public:
  Parser(clang::DiagnosticsEngine &diagnosticsEngine,
         clang::SourceManager &sourceManager,
         std::shared_ptr<SourceFile> source);

  /// Parse the whole content of the file ('stored-definition').
  std::unique_ptr<ast::modelica::StoredDefinition> parseStoredDefinition();

  /// Parse a 'class-definition', without the prefixes of elements.
  std::unique_ptr<ast::modelica::ClassDefinition> parseClassDefinition();

  /// Parse an 'expression'.
  std::unique_ptr<ast::modelica::Expression> parseExpression();

  /// Parse a 'some-equation'.
  std::unique_ptr<ast::modelica::Equation> parseEquation();

  /// Parse a 'statement'.
  std::unique_ptr<ast::modelica::Statement> parseStatement();

  /// Parse a 'modification'.
  std::unique_ptr<ast::modelica::Modification> parseModification();

  /// Whether the whole input has been consumed.
  bool isAtEnd() const;

private:
  using TK = TokenKind;

  //===-------------------------------------------------------------------===//
  // Token handling
  //===-------------------------------------------------------------------===//

  const Token &peek(size_t offset = 0) const;
  bool check(TokenKind kind, size_t offset = 0) const;
  const Token &advance();
  bool accept(TokenKind kind);
  bool expect(TokenKind kind);

  /// Location spanning from the token with the given index to the last
  /// consumed token.
  SourceRange rangeFrom(size_t startIndex) const;

  template <typename T>
  std::unique_ptr<T> makeNode(size_t startIndex) {
    return std::make_unique<T>(rangeFrom(startIndex));
  }

  template <typename T>
  void finalize(T &node, size_t startIndex) {
    node.setLocation(rangeFrom(startIndex));
  }

  //===-------------------------------------------------------------------===//
  // Diagnostics
  //===-------------------------------------------------------------------===//

  clang::SourceLocation convertLocation(const SourceRange &location) const;
  void emitError(const SourceRange &location, llvm::StringRef message);
  void emitUnexpectedTokenError(llvm::StringRef expected);

  //===-------------------------------------------------------------------===//
  // Classes
  //===-------------------------------------------------------------------===//

  bool isClassDefinitionStart() const;
  bool parseClassPrefixes(ast::modelica::ClassDefinition &classDefinition);

  std::unique_ptr<ast::modelica::ClassDefinition>
  parseClassDefinition(const ast::modelica::ElementPrefixes &prefixes);

  bool parseShortClassSpecifier(ast::modelica::ClassDefinition &definition);
  bool parseComposition(ast::modelica::ClassDefinition &definition);
  bool isElementListEnd() const;
  bool parseElementList(ast::modelica::ClassDefinition &definition,
                        bool isProtected);
  std::unique_ptr<ast::modelica::Node> parseElement(bool isProtected);
  std::unique_ptr<ast::modelica::ImportClause> parseImportClause();
  std::unique_ptr<ast::modelica::ExternalClause> parseExternalClause();

  //===-------------------------------------------------------------------===//
  // Extends and components
  //===-------------------------------------------------------------------===//

  std::unique_ptr<ast::modelica::ExtendsClause> parseExtendsClause();

  std::unique_ptr<ast::modelica::ConstrainingClause> parseConstrainingClause();

  bool parseTypePrefix(ast::modelica::TypePrefix &prefix);

  std::unique_ptr<ast::modelica::ComponentClause>
  parseComponentClause(const ast::modelica::ElementPrefixes &prefixes,
                       bool singleDeclaration);

  std::unique_ptr<ast::modelica::ComponentDeclaration>
  parseComponentDeclaration(bool allowCondition);

  //===-------------------------------------------------------------------===//
  // Modifications
  //===-------------------------------------------------------------------===//

  std::unique_ptr<ast::modelica::ClassModification>
  parseClassModification(bool allowInheritanceModifications = false);

  std::unique_ptr<ast::modelica::Argument>
  parseArgument(bool allowInheritanceModifications);

  std::unique_ptr<ast::modelica::Argument>
  parseElementReplaceable(bool isRedeclare, bool isEach, bool isFinal);

  std::unique_ptr<ast::modelica::Argument> parseInheritanceModification();

  //===-------------------------------------------------------------------===//
  // Equations and statements
  //===-------------------------------------------------------------------===//

  std::unique_ptr<ast::modelica::EquationSection> parseEquationSection();
  std::unique_ptr<ast::modelica::AlgorithmSection> parseAlgorithmSection();

  bool isEquationListEnd() const;

  bool parseEquationList(
      ast::modelica::Children<ast::modelica::Equation> &equations,
      llvm::ArrayRef<TokenKind> terminators);

  bool parseStatementList(
      ast::modelica::Children<ast::modelica::Statement> &statements,
      llvm::ArrayRef<TokenKind> terminators);

  std::unique_ptr<ast::modelica::Equation> parseIfEquation();
  std::unique_ptr<ast::modelica::Equation> parseForEquation();
  std::unique_ptr<ast::modelica::Equation> parseWhenEquation();
  std::unique_ptr<ast::modelica::Equation> parseConnectEquation();

  std::unique_ptr<ast::modelica::Statement> parseIfStatement();
  std::unique_ptr<ast::modelica::Statement> parseForStatement();
  std::unique_ptr<ast::modelica::Statement> parseWhileStatement();
  std::unique_ptr<ast::modelica::Statement> parseWhenStatement();

  bool parseForIndices(ast::modelica::Children<ast::modelica::ForIndex> &indices);
  std::unique_ptr<ast::modelica::ForIndex> parseForIndex();

  //===-------------------------------------------------------------------===//
  // Expressions
  //===-------------------------------------------------------------------===//

  std::unique_ptr<ast::modelica::Expression> parseSimpleExpression();
  std::unique_ptr<ast::modelica::Expression> parseLogicalExpression();
  std::unique_ptr<ast::modelica::Expression> parseLogicalTerm();
  std::unique_ptr<ast::modelica::Expression> parseLogicalFactor();
  std::unique_ptr<ast::modelica::Expression> parseRelation();
  std::unique_ptr<ast::modelica::Expression> parseArithmeticExpression();
  std::unique_ptr<ast::modelica::Expression> parseTerm();
  std::unique_ptr<ast::modelica::Expression> parseFactor();
  std::unique_ptr<ast::modelica::Expression> parsePrimary();
  std::unique_ptr<ast::modelica::Expression> parseNumber();
  std::unique_ptr<ast::modelica::Expression> parseParenthesized();
  std::unique_ptr<ast::modelica::Expression> parseArrayConcatenation();
  std::unique_ptr<ast::modelica::Expression> parseArrayConstructor();

  std::unique_ptr<ast::modelica::ComponentReference> parseComponentReference();

  std::unique_ptr<ast::modelica::Call>
  parseFunctionCallArgs(std::unique_ptr<ast::modelica::ComponentReference> callee,
                        size_t startIndex);

  bool isNamedArgumentStart() const;
  std::unique_ptr<ast::modelica::NamedArgument> parseNamedArgument();
  std::unique_ptr<ast::modelica::Expression> parseFunctionArgument();

  bool parseArraySubscripts(
      ast::modelica::Children<ast::modelica::Expression> &subscripts);

  std::unique_ptr<ast::modelica::Expression> parseSubscript();

  //===-------------------------------------------------------------------===//
  // Names and descriptions
  //===-------------------------------------------------------------------===//

  std::optional<std::string> parseIdentifier();
  std::optional<ast::modelica::Name> parseName();
  std::optional<ast::modelica::Name> parseTypeSpecifier();
  std::optional<std::string> parseDescriptionString();
  bool parseDescription(ast::modelica::Description &description);
  std::unique_ptr<ast::modelica::ClassModification> parseAnnotation();

private:
  clang::DiagnosticsEngine *diagnosticsEngine;
  clang::SourceManager *sourceManager;
  std::vector<Token> tokens;
  size_t index = 0;

  /// Number of array subscripts being parsed, used to accept 'end' as
  /// expression only within subscripts.
  unsigned subscriptsDepth = 0;
};
} // namespace marco::parser::modelica

#endif // MARCO_PARSER_MODELICA_PARSER_H
