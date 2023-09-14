#ifndef MARCO_PARSER_PARSER_H
#define MARCO_PARSER_PARSER_H

#include "marco/Diagnostic/Diagnostic.h"
#include "marco/Diagnostic/Location.h"
#include "marco/Parser/ModelicaStateMachine.h"
#include "marco/AST/AST.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace marco::parser
{
	/// The parser encapsulates the lexer but not the memory where the string we
	/// are reading is held. It exposes parts of the grammatical rules that are
	/// available in the grammar (can be found at page ~ 265 of the 3.4 doc).
  class Parser
  {
    public:
      template<typename T>
      class ValueWrapper
      {
        public:
          ValueWrapper(SourceRange location, T value)
            : location(std::move(location)), value(std::move(value))
          {
          }

          SourceRange getLocation()
          {
            return location;
          }

          T& getValue()
          {
            return value;
          }

          const T& getValue() const
          {
            return value;
          }

        private:
          SourceRange location;
          T value;
      };

      Parser(diagnostic::DiagnosticEngine& diagnostics, std::shared_ptr<SourceFile> source);

      std::optional<std::unique_ptr<ast::ASTNode>> parseRoot();

      /// Parse a boolean value.
      std::optional<ValueWrapper<bool>> parseBoolValue();

      /// Parse an integer value.
      std::optional<ValueWrapper<int64_t>> parseIntValue();

      /// Parse a floating point value.
      std::optional<ValueWrapper<double>> parseFloatValue();

      /// Parse a string.
      std::optional<ValueWrapper<std::string>> parseString();

      /// Parse the name of an identifier.
      std::optional<ValueWrapper<std::string>> parseIdentifier();

      /// Parse the 'class-definition' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseClassDefinition();

      /// Parse the 'modification' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseModification();

      /// Parse the 'class-modification' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseClassModification();

      /// Parse the 'argument-list' symbol.
      std::optional<std::vector<std::unique_ptr<ast::ASTNode>>> parseArgumentList();

      /// Parse the 'argument' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseArgument();

      /// Parse the 'element-modification' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseElementModification(bool each, bool final);

      /// Parse the 'element-redeclaration' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseElementRedeclaration();

      /// Parse the 'element-replaceable' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseElementReplaceable(bool each, bool final);

      /// Parse the 'algorithm-section' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseAlgorithmSection();

      /// Parse the 'equation' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseEquation();

      /// Parse the 'statement' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseStatement();

      /// Parse the 'if-statement' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseIfStatement();

      /// Parse the 'for-statement'.
      std::optional<std::unique_ptr<ast::ASTNode>> parseForStatement();

      /// Parse the 'while-statement'.
      std::optional<std::unique_ptr<ast::ASTNode>> parseWhileStatement();

      /// Parse the 'when-statement'.
      std::optional<std::unique_ptr<ast::ASTNode>> parseWhenStatement();

      /// Parse the 'expression' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseExpression();

      /// Parse the 'simple-expression' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseSimpleExpression();

      /// Parse the 'logical-expression' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseLogicalExpression();

      /// Parse the 'logical-term' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseLogicalTerm();

      /// Parse the 'logical-factor' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseLogicalFactor();

      /// Parse the 'relation' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseRelation();

      /// Parse the 'relational-operator' symbol.
      std::optional<ast::OperationKind> parseRelationalOperator();

      /// Parse the 'arithmetic-expression' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseArithmeticExpression();

      /// Parse the 'add-operator' symbol.
      std::optional<ast::OperationKind> parseAddOperator();

      /// Parse the 'term' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseTerm();

      /// Parse the mul-operator' symbol.
      std::optional<ast::OperationKind> parseMulOperator();

      /// Parse the 'factor' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseFactor();

      /// Parse the 'primary' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parsePrimary();

      /// Parse the 'component-reference' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseComponentReference();

      std::optional<std::unique_ptr<ast::ASTNode>>
      parseComponentReferenceEntry();

      /// Parse the 'function-call-args' symbol.
      std::optional<ValueWrapper<std::vector<std::unique_ptr<ast::ASTNode>>>> parseFunctionCallArgs();

      /// Parse the 'function-arguments' symbol.
      std::optional<std::vector<std::unique_ptr<ast::ASTNode>>> parseFunctionArguments();

      /// Parse the 'function-arguments-non-first' symbol.
      std::optional<std::vector<std::unique_ptr<ast::ASTNode>>> parseFunctionArgumentsNonFirst();

      /// Parse the 'array-arguments' symbol.
      std::optional<std::pair<std::vector<std::unique_ptr<ast::ASTNode>>, std::vector<std::unique_ptr<ast::ASTNode>>>> parseArrayArguments();

      /// Parse the 'array-arguments-non-first' symbol.
      std::optional<std::vector<std::unique_ptr<ast::ASTNode>>> parseArrayArgumentsNonFirst();

      /// Parse the 'named-arguments' symbol.
      std::optional<std::vector<std::unique_ptr<ast::ASTNode>>> parseNamedArguments();

      /// Parse the 'named-argument' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseNamedArgument();

      /// Parse the 'function-argument' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseFunctionArgument();

      /// Parse the 'output-expression-list' symbol.
      std::optional<std::vector<std::unique_ptr<ast::ASTNode>>> parseOutputExpressionList();

      /// Parse the 'array-subscripts' symbol.
      std::optional<ValueWrapper<std::vector<std::unique_ptr<ast::ASTNode>>>> parseArraySubscripts();

      /// Parse the 'subscript' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseSubscript();

      /// Parse the 'annotation' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseAnnotation();

    private:
      std::optional<std::unique_ptr<ast::ASTNode>> parseEquationsBlock();

      std::optional<std::vector<std::unique_ptr<ast::ASTNode>>> parseForEquations();

      /// Parse the 'for-index' symbol.
      std::optional<std::unique_ptr<ast::ASTNode>> parseForIndex();

      std::optional<std::vector<std::unique_ptr<ast::ASTNode>>> parseElementList(bool publicSection);

      std::optional<std::unique_ptr<ast::ASTNode>> parseElement(bool publicSection);

      std::optional<std::unique_ptr<ast::ASTNode>> parseTypePrefix();

      std::optional<std::unique_ptr<ast::ASTNode>> parseVariableType();

      std::optional<std::unique_ptr<ast::ASTNode>> parseArrayDimension();

      std::optional<std::unique_ptr<ast::ASTNode>> parseTermModification();

    private:
      /// Read the next token.
      void advance();

      /// Accept the current token and move to the next one, but only
      /// if the current token is the one being expected. The function
      /// returns whether the expected token has been found.
      template<TokenKind Kind>
      bool accept()
      {
        if (tokens.front().getKind() == Kind) {
          advance();
          return true;
        }

        return false;
      }

    private:
      diagnostic::DiagnosticEngine* diagnostics;
      Lexer<ModelicaStateMachine> lexer;
      llvm::SmallVector<Token, 2> tokens{2, Token(TokenKind::Begin)};
  };
}

#endif // MARCO_PARSER_PARSER_H
