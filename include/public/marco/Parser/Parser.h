#ifndef MARCO_PARSER_PARSER_H
#define MARCO_PARSER_PARSER_H

#include "marco/Diagnostic/Diagnostic.h"
#include "marco/Diagnostic/Location.h"
#include "marco/Parser/ModelicaStateMachine.h"
#include "marco/AST/AST.h"
#include "llvm/ADT/Optional.h"

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

      Parser(diagnostic::DiagnosticEngine& diagnostics, llvm::StringRef file, const char* source);
      Parser(diagnostic::DiagnosticEngine& diagnostics, const std::string& source);
      Parser(diagnostic::DiagnosticEngine& diagnostics, const char* source);

      /// Parse a boolean value.
      llvm::Optional<ValueWrapper<bool>> parseBoolValue();

      /// Parse an integer value.
      llvm::Optional<ValueWrapper<int64_t>> parseIntValue();

      /// Parse a floating point value.
      llvm::Optional<ValueWrapper<double>> parseFloatValue();

      /// Parse a string.
      llvm::Optional<ValueWrapper<std::string>> parseString();

      /// Parse the name of an identifier.
      llvm::Optional<ValueWrapper<std::string>> parseIdentifier();

      /// Parse the 'class-definition' symbol.
      llvm::Optional<std::unique_ptr<ast::Class>> parseClassDefinition();

      /// Parse the 'modification' symbol.
      llvm::Optional<std::unique_ptr<ast::Modification>> parseModification();

      /// Parse the 'class-modification' symbol.
      llvm::Optional<std::unique_ptr<ast::ClassModification>> parseClassModification();

      /// Parse the 'argument-list' symbol.
      llvm::Optional<std::vector<std::unique_ptr<ast::Argument>>> parseArgumentList();

      /// Parse the 'argument' symbol.
      llvm::Optional<std::unique_ptr<ast::Argument>> parseArgument();

      /// Parse the 'element-modification' symbol.
      llvm::Optional<std::unique_ptr<ast::Argument>> parseElementModification(bool each, bool final);

      /// Parse the 'element-redeclaration' symbol.
      llvm::Optional<std::unique_ptr<ast::Argument>> parseElementRedeclaration();

      /// Parse the 'element-replaceable' symbol.
      llvm::Optional<std::unique_ptr<ast::Argument>> parseElementReplaceable(bool each, bool final);

      /// Parse the 'algorithm-section' symbol.
      llvm::Optional<std::unique_ptr<ast::Algorithm>> parseAlgorithmSection();

      /// Parse the 'equation' symbol.
      llvm::Optional<std::unique_ptr<ast::Equation>> parseEquation();

      /// Parse the 'statement' symbol.
      llvm::Optional<std::unique_ptr<ast::Statement>> parseStatement();

      /// Parse the 'if-statement' symbol.
      llvm::Optional<std::unique_ptr<ast::Statement>> parseIfStatement();

      /// Parse the 'for-statement'.
      llvm::Optional<std::unique_ptr<ast::Statement>> parseForStatement();

      /// Parse the 'while-statement'.
      llvm::Optional<std::unique_ptr<ast::Statement>> parseWhileStatement();

      /// Parse the 'when-statement'.
      llvm::Optional<std::unique_ptr<ast::Statement>> parseWhenStatement();

      /// Parse the 'expression' symbol.
      llvm::Optional<std::unique_ptr<ast::Expression>> parseExpression();

      /// Parse the 'simple-expression' symbol.
      llvm::Optional<std::unique_ptr<ast::Expression>> parseSimpleExpression();

      /// Parse the 'logical-expression' symbol.
      llvm::Optional<std::unique_ptr<ast::Expression>> parseLogicalExpression();

      /// Parse the 'logical-term' symbol.
      llvm::Optional<std::unique_ptr<ast::Expression>> parseLogicalTerm();

      /// Parse the 'logical-factor' symbol.
      llvm::Optional<std::unique_ptr<ast::Expression>> parseLogicalFactor();

      /// Parse the 'relation' symbol.
      llvm::Optional<std::unique_ptr<ast::Expression>> parseRelation();

      /// Parse the 'relational-operator' symbol.
      llvm::Optional<ast::OperationKind> parseRelationalOperator();

      /// Parse the 'arithmetic-expression' symbol.
      llvm::Optional<std::unique_ptr<ast::Expression>> parseArithmeticExpression();

      /// Parse the 'add-operator' symbol.
      llvm::Optional<ast::OperationKind> parseAddOperator();

      /// Parse the 'term' symbol.
      llvm::Optional<std::unique_ptr<ast::Expression>> parseTerm();

      /// Parse the mul-operator' symbol.
      llvm::Optional<ast::OperationKind> parseMulOperator();

      /// Parse the 'factor' symbol.
      llvm::Optional<std::unique_ptr<ast::Expression>> parseFactor();

      /// Parse the 'primary' symbol.
      llvm::Optional<std::unique_ptr<ast::Expression>> parsePrimary();

      /// Parse the 'component-reference' symbol.
      llvm::Optional<std::unique_ptr<ast::Expression>> parseComponentReference();

      /// Parse the 'function-call-args' symbol.
      llvm::Optional<ValueWrapper<std::vector<std::unique_ptr<ast::Expression>>>> parseFunctionCallArgs();

      /// Parse the 'function-arguments' symbol.
      llvm::Optional<std::vector<std::unique_ptr<ast::Expression>>> parseFunctionArguments();

      /// Parse the 'function-arguments-non-first' symbol.
      llvm::Optional<std::vector<std::unique_ptr<ast::Expression>>> parseFunctionArgumentsNonFirst();

      /// Parse the 'array-arguments' symbol.
      llvm::Optional<std::vector<std::unique_ptr<ast::Expression>>> parseArrayArguments();

      /// Parse the 'array-arguments-non-first' symbol.
      llvm::Optional<std::vector<std::unique_ptr<ast::Expression>>> parseArrayArgumentsNonFirst();

      /// Parse the 'function-argument' symbol.
      llvm::Optional<std::unique_ptr<ast::Expression>> parseFunctionArgument();

      /// Parse the 'output-expression-list' symbol.
      llvm::Optional<std::vector<std::unique_ptr<ast::Expression>>> parseOutputExpressionList();

      /// Parse the 'array-subscripts' symbol.
      llvm::Optional<ValueWrapper<std::vector<std::unique_ptr<ast::Expression>>>> parseArraySubscripts();

      /// Parse the 'subscript' symbol.
      llvm::Optional<std::unique_ptr<ast::Expression>> parseSubscript();

      /// Parse the 'annotation' symbol.
      llvm::Optional<std::unique_ptr<ast::Annotation>> parseAnnotation();

    private:
      llvm::Optional<std::unique_ptr<ast::EquationsBlock>> parseEquationsBlock();

      llvm::Optional<std::vector<std::unique_ptr<ast::ForEquation>>> parseForEquations();

      llvm::Optional<std::unique_ptr<ast::Induction>> parseInduction();

      llvm::Optional<std::vector<std::unique_ptr<ast::Member>>> parseElementList(bool publicSection);

      llvm::Optional<std::unique_ptr<ast::Member>> parseElement(bool publicSection);

      llvm::Optional<ast::TypePrefix> parseTypePrefix();

      llvm::Optional<ast::Type> parseTypeSpecifier();

      llvm::Optional<std::unique_ptr<ast::Expression>> parseTermModification();

      private:
      /// Read the next token.
      void next();

      /// Accept the current token and move to the next one, but only
      /// if the current token is the one being expected. The function
      /// returns whether the expected token has been found.
      template<Token t>
      bool accept()
      {
        if (current == t) {
          next();
          return true;
        }

        return false;
      }

    private:
      diagnostic::DiagnosticEngine* diagnostics;
      const char* source;
      Lexer<ModelicaStateMachine> lexer;
      Token current;
  };
}

#endif // MARCO_PARSER_PARSER_H
