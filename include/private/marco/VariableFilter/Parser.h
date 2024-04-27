#ifndef MARCO_VARIABLEFILTER_PARSER_H
#define MARCO_VARIABLEFILTER_PARSER_H

#include "marco/VariableFilter/AST.h"
#include "marco/VariableFilter/LexerStateMachine.h"
#include "marco/VariableFilter/VariableFilter.h"
#include "marco/Parser/Lexer.h"
#include "marco/Parser/Location.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringRef.h"

namespace marco::vf
{
  class Parser
  {
    public:
      Parser(
        VariableFilter& vf,
        clang::DiagnosticsEngine& diagnosticsEngine,
        clang::SourceManager& sourceManager,
        std::shared_ptr<SourceFile> source);

      bool run();

      bool parseListElement();

      std::optional<VariableExpression> parseVariableExpression();

      std::optional<ArrayExpression> parseArray(VariableExpression variable);

      std::optional<Range> parseArrayRange();

      std::optional<int> parseArrayIndex();

      std::optional<DerivativeExpression> parseDer();

      std::optional<RegexExpression> parseRegex();

    private:
      /// Move to the next token.
      void advance();

      /// Check if the first lookahead has the requested type.
      /// If it does, shift the tokens. Otherwise, keep the current state.
      template<TokenKind Kind>
      bool accept()
      {
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

      /// @name Diagnostics
      /// {

      clang::SourceLocation convertLocation(const SourceRange& location) const;

      void emitUnexpectedTokenError(const Token& found, TokenKind expected);

      void emitEmptyRegexError(const SourceRange& location);

      void emitInvalidRegexError(const SourceRange& location);

      /// }

    private:
      VariableFilter* vf;
      clang::DiagnosticsEngine* diagnosticsEngine;
      clang::SourceManager* sourceManager;
      Lexer<LexerStateMachine> lexer;
      Token token{TokenKind::Begin};
      llvm::SmallVector<Token, 1> lookahead{1, Token(TokenKind::Begin)};
  };
}

#endif // MARCO_VARIABLEFILTER_PARSER_H
