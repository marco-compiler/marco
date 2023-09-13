#ifndef MARCO_VARIABLEFILTER_PARSER_H
#define MARCO_VARIABLEFILTER_PARSER_H

#include "marco/Diagnostic/Diagnostic.h"
#include "marco/VariableFilter/AST.h"
#include "marco/VariableFilter/LexerStateMachine.h"
#include "marco/VariableFilter/VariableFilter.h"
#include "marco/Diagnostic/Location.h"
#include "marco/Parser/Lexer.h"
#include "llvm/ADT/StringRef.h"

namespace marco::vf
{
  class Parser
  {
    public:
      Parser(VariableFilter& vf, diagnostic::DiagnosticEngine* diagnostics, std::shared_ptr<SourceFile> file);

      bool run();

      bool token();

      std::optional<DerivativeExpression> der();

      std::optional<RegexExpression> regex();

      std::optional<VariableExpression> identifier();

      std::optional<ArrayExpression> array(VariableExpression variable);

      std::optional<Range> arrayRange();

    private:
      /// Read the next token.
      void next();

      /// Regular accept: if the current token is t then the next one will be read
      /// and true will be returned, else false.
      bool accept(Token t);

      /// fancy overloads if you know at compile time
      /// which token you want.
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
      VariableFilter* vf;
      diagnostic::DiagnosticEngine* diagnostics;
      Lexer<LexerStateMachine> lexer;
      Token current;
  };
}

#endif // MARCO_VARIABLEFILTER_PARSER_H
