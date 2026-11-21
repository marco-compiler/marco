#ifndef MARCO_PARSER_MODELICA_PARSER_H
#define MARCO_PARSER_MODELICA_PARSER_H

#include "marco/AST/Modelica/AST.h"
#include "marco/Lexer/Lexer.h"
#include "marco/Parser/Modelica/StateMachine.h"
#include "marco/Parser/Modelica/Token.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace marco::parser::modelica {
/// Recursive descent parser for the Modelica language, following the grammar
/// of the language specification. Parsing stops at the first error, which is
/// reported through the diagnostics engine. The parse methods return a null
/// pointer or false in case of error.
class Parser {
public:
  Parser(clang::DiagnosticsEngine &diagnosticsEngine,
         clang::SourceManager &sourceManager,
         std::shared_ptr<SourceFile> source);

  /// Parse the 'stored-definition' symbol, that is the whole content of the
  /// file.
  std::unique_ptr<ast::modelica::StoredDefinition> parseStoredDefinition();

  /// Parse the 'class-definition' symbol, without the prefixes of elements.
  std::unique_ptr<ast::modelica::ClassDefinition> parseClassDefinition();

  /// Parse the 'expression' symbol.
  std::unique_ptr<ast::modelica::Expression> parseExpression();

  /// Parse the 'some-equation' symbol.
  std::unique_ptr<ast::modelica::Equation> parseEquation();

  /// Parse the 'statement' symbol.
  std::unique_ptr<ast::modelica::Statement> parseStatement();

  /// Parse the 'modification' symbol.
  std::unique_ptr<ast::modelica::Modification> parseModification();

  /// Whether the whole input has been consumed.
  bool isAtEnd() const;

private:
  //===-------------------------------------------------------------------===//
  // Token handling
  //===-------------------------------------------------------------------===//

  /// Shift the lookahead by one token, scanning the next one. The end of the
  /// file and the errors are repeated instead of scanning further.
  void shiftLookahead();

  /// Get the current token (offset 0) or the following one (offset 1).
  const Token &peek(size_t offset = 0) const;
  bool check(TokenKind kind, size_t offset = 0) const;
  Token advance();
  bool accept(TokenKind kind);
  bool expect(TokenKind kind);

  /// Location spanning from the given starting token to the last consumed
  /// one. At least one token must have been consumed since the start.
  SourceRange rangeFrom(const SourceRange &start) const;

  /// Create a node located at its starting token, whose location is then
  /// extended by 'finalize'.
  template <typename T>
  std::unique_ptr<T> makeNode(const SourceRange &start) {
    return std::make_unique<T>(start);
  }

  template <typename T>
  void finalize(T &node, const SourceRange &start) {
    node.setLocation(rangeFrom(start));
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

  /// Get whether the current token starts a 'class-definition' symbol.
  bool isClassDefinitionStart() const;

  /// Parse the 'class-prefixes' symbol.
  bool parseClassPrefixes(ast::modelica::ClassDefinition &classDefinition);

  /// Parse the 'class-definition' symbol, preceded by the given prefixes of
  /// the element.
  std::unique_ptr<ast::modelica::ClassDefinition>
  parseClassDefinition(const ast::modelica::ElementPrefixes &prefixes);

  /// Parse the 'short-class-specifier' symbol of non-enumerations, from the
  /// 'base-prefix' symbol on.
  bool parseShortClassSpecifier(ast::modelica::ClassDefinition &definition);

  /// Parse the 'composition' symbol.
  bool parseComposition(ast::modelica::ClassDefinition &definition);

  /// Get whether the current token ends an 'element-list' symbol.
  bool isElementListEnd() const;

  /// Parse the 'element-list' symbol, whose elements are public or
  /// protected.
  bool parseElementList(ast::modelica::ClassDefinition &definition,
                        bool isProtected);

  /// Parse the 'element' symbol.
  std::unique_ptr<ast::modelica::Node> parseElement(bool isProtected);

  /// Parse the 'import-clause' symbol.
  std::unique_ptr<ast::modelica::ImportClause> parseImportClause();

  /// Parse the external clause of the 'composition' symbol, starting from
  /// 'external'.
  std::unique_ptr<ast::modelica::ExternalClause> parseExternalClause();

  //===-------------------------------------------------------------------===//
  // Extends and components
  //===-------------------------------------------------------------------===//

  /// Parse the 'extends-clause' symbol.
  std::unique_ptr<ast::modelica::ExtendsClause> parseExtendsClause();

  /// Parse the 'constraining-clause' symbol.
  std::unique_ptr<ast::modelica::ConstrainingClause> parseConstrainingClause();

  /// Parse the 'type-prefix' symbol.
  bool parseTypePrefix(ast::modelica::TypePrefix &prefix);

  /// Parse the 'component-clause' symbol, or the 'component-clause1' one if
  /// 'singleDeclaration' is true.
  std::unique_ptr<ast::modelica::ComponentClause>
  parseComponentClause(const ast::modelica::ElementPrefixes &prefixes,
                       bool singleDeclaration);

  /// Parse the 'component-declaration' symbol if 'allowCondition' is true,
  /// or the 'component-declaration1' one otherwise.
  std::unique_ptr<ast::modelica::ComponentDeclaration>
  parseComponentDeclaration(bool allowCondition);

  //===-------------------------------------------------------------------===//
  // Modifications
  //===-------------------------------------------------------------------===//

  /// Parse the 'class-modification' symbol. If
  /// 'allowInheritanceModifications' is true, the arguments can also be
  /// 'inheritance-modification' symbols.
  std::unique_ptr<ast::modelica::ClassModification>
  parseClassModification(bool allowInheritanceModifications = false);

  /// Parse the 'argument' symbol, or the 'inheritance-modification' one if
  /// allowed.
  std::unique_ptr<ast::modelica::Argument>
  parseArgument(bool allowInheritanceModifications);

  /// Parse the 'element-redeclaration' symbol if 'isRedeclare' is true, or
  /// the 'element-replaceable' one otherwise, after the 'redeclare', 'each'
  /// and 'final' prefixes.
  std::unique_ptr<ast::modelica::Argument>
  parseElementReplaceable(bool isRedeclare, bool isEach, bool isFinal);

  /// Parse the 'inheritance-modification' symbol.
  std::unique_ptr<ast::modelica::Argument> parseInheritanceModification();

  //===-------------------------------------------------------------------===//
  // Equations and statements
  //===-------------------------------------------------------------------===//

  /// Parse the 'equation-section' symbol.
  std::unique_ptr<ast::modelica::EquationSection> parseEquationSection();

  /// Parse the 'algorithm-section' symbol.
  std::unique_ptr<ast::modelica::AlgorithmSection> parseAlgorithmSection();

  /// Get whether the current token ends the list of equations of an
  /// 'equation-section' symbol.
  bool isEquationListEnd() const;

  /// Parse a sequence of 'some-equation' symbols, each followed by ';', up to
  /// one of the given tokens.
  bool
  parseEquationList(ast::modelica::Children<ast::modelica::Equation> &equations,
                    llvm::ArrayRef<TokenKind> terminators);

  /// Parse a sequence of 'statement' symbols, each followed by ';', up to
  /// one of the given tokens.
  bool parseStatementList(
      ast::modelica::Children<ast::modelica::Statement> &statements,
      llvm::ArrayRef<TokenKind> terminators);

  /// Parse the 'if-equation' symbol.
  std::unique_ptr<ast::modelica::Equation> parseIfEquation();

  /// Parse the 'for-equation' symbol.
  std::unique_ptr<ast::modelica::Equation> parseForEquation();

  /// Parse the 'when-equation' symbol.
  std::unique_ptr<ast::modelica::Equation> parseWhenEquation();

  /// Parse the 'connect-equation' symbol.
  std::unique_ptr<ast::modelica::Equation> parseConnectEquation();

  /// Parse the 'if-statement' symbol.
  std::unique_ptr<ast::modelica::Statement> parseIfStatement();

  /// Parse the 'for-statement' symbol.
  std::unique_ptr<ast::modelica::Statement> parseForStatement();

  /// Parse the 'while-statement' symbol.
  std::unique_ptr<ast::modelica::Statement> parseWhileStatement();

  /// Parse the 'when-statement' symbol.
  std::unique_ptr<ast::modelica::Statement> parseWhenStatement();

  /// Parse the 'for-indices' symbol.
  bool
  parseForIndices(ast::modelica::Children<ast::modelica::ForIndex> &indices);

  /// Parse the 'for-index' symbol.
  std::unique_ptr<ast::modelica::ForIndex> parseForIndex();

  //===-------------------------------------------------------------------===//
  // Expressions
  //===-------------------------------------------------------------------===//

  /// Parse the 'simple-expression' symbol.
  std::unique_ptr<ast::modelica::Expression> parseSimpleExpression();

  /// Parse the 'logical-expression' symbol.
  std::unique_ptr<ast::modelica::Expression> parseLogicalExpression();

  /// Parse the 'logical-term' symbol.
  std::unique_ptr<ast::modelica::Expression> parseLogicalTerm();

  /// Parse the 'logical-factor' symbol.
  std::unique_ptr<ast::modelica::Expression> parseLogicalFactor();

  /// Parse the 'relation' symbol.
  std::unique_ptr<ast::modelica::Expression> parseRelation();

  /// Parse the 'arithmetic-expression' symbol.
  std::unique_ptr<ast::modelica::Expression> parseArithmeticExpression();

  /// Parse the 'term' symbol.
  std::unique_ptr<ast::modelica::Expression> parseTerm();

  /// Parse the 'factor' symbol.
  std::unique_ptr<ast::modelica::Expression> parseFactor();

  /// Parse the 'primary' symbol.
  std::unique_ptr<ast::modelica::Expression> parsePrimary();

  /// Parse the 'UNSIGNED-NUMBER' symbol.
  std::unique_ptr<ast::modelica::Expression> parseNumber();

  /// Parse the '"(" output-expression-list ")"' alternative of the 'primary'
  /// symbol, with the optional array subscripts or member access.
  std::unique_ptr<ast::modelica::Expression> parseParenthesized();

  /// Parse the '"[" expression-list { ";" expression-list } "]"'
  /// alternative of the 'primary' symbol.
  std::unique_ptr<ast::modelica::Expression> parseArrayConcatenation();

  /// Parse the '"{" array-arguments "}"' alternative of the 'primary' symbol.
  std::unique_ptr<ast::modelica::Expression> parseArrayConstructor();

  /// Parse the 'component-reference' symbol.
  std::unique_ptr<ast::modelica::ComponentReference> parseComponentReference();

  /// Parse the 'function-call-args' symbol of a call to the given function.
  std::unique_ptr<ast::modelica::Call> parseFunctionCallArgs(
      std::unique_ptr<ast::modelica::ComponentReference> callee,
      const SourceRange &start);

  /// Get whether the current token starts a 'named-argument' symbol.
  bool isNamedArgumentStart() const;

  /// Parse the 'named-argument' symbol.
  std::unique_ptr<ast::modelica::NamedArgument> parseNamedArgument();

  /// Parse the 'function-argument' symbol.
  std::unique_ptr<ast::modelica::Expression> parseFunctionArgument();

  /// Parse the 'array-subscripts' symbol.
  bool parseArraySubscripts(
      ast::modelica::Children<ast::modelica::Expression> &subscripts);

  /// Parse the 'subscript' symbol.
  std::unique_ptr<ast::modelica::Expression> parseSubscript();

  //===-------------------------------------------------------------------===//
  // Names and descriptions
  //===-------------------------------------------------------------------===//

  /// Parse the 'IDENT' symbol.
  std::optional<std::string> parseIdentifier();

  /// Parse the 'name' symbol.
  std::optional<ast::modelica::Name> parseName();

  /// Parse the 'type-specifier' symbol.
  std::optional<ast::modelica::Name> parseTypeSpecifier();

  /// Parse the 'description-string' symbol.
  std::optional<std::string> parseDescriptionString();

  /// Parse the 'description' symbol.
  bool parseDescription(ast::modelica::Description &description);

  /// Parse the 'annotation-clause' symbol.
  std::unique_ptr<ast::modelica::ClassModification> parseAnnotation();

private:
  clang::DiagnosticsEngine *diagnosticsEngine;
  clang::SourceManager *sourceManager;
  lexer::Lexer<StateMachine> lexer;

  /// The current token and the following one. The end of the file and the
  /// errors are never consumed.
  llvm::SmallVector<Token, 2> lookahead{2, Token(TokenKind::Begin)};

  SourceRange lastConsumedLocation = SourceRange::unknown();
};
} // namespace marco::parser::modelica

#endif // MARCO_PARSER_MODELICA_PARSER_H
