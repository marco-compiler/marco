#ifndef MARCO_VARIABLEFILTER_PARSER_H
#define MARCO_VARIABLEFILTER_PARSER_H

#include "marco/VariableFilter/AST.h"
#include "marco/VariableFilter/LexerStateMachine.h"
#include "marco/VariableFilter/VariableFilter.h"
#include "marco/Utils/Lexer.h"
#include "marco/Utils/SourcePosition.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace marco::vf
{
  class Parser
  {
    public:
      Parser(VariableFilter& vf, llvm::StringRef source);

      llvm::Error run();

      llvm::Error token();

      llvm::Expected<DerivativeExpression> der();

      llvm::Expected<RegexExpression> regex();

      llvm::Expected<VariableExpression> identifier();

      llvm::Expected<ArrayExpression> array(VariableExpression variable);

      llvm::Expected<Range> arrayRange();

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

      llvm::Expected<bool> expect(Token t);

      void updateTokenSourceRange();

    private:
      VariableFilter* vf;
      Lexer<LexerStateMachine> lexer;
      Token current;
      SourceRange tokenRange;
  };
}

#endif // MARCO_VARIABLEFILTER_PARSER_H
