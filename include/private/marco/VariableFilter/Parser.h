#ifndef MARCO_VARIABLEFILTER_PARSER_H
#define MARCO_VARIABLEFILTER_PARSER_H

#include "marco/VariableFilter/AST.h"
#include "marco/VariableFilter/LexerStateMachine.h"
#include "marco/VariableFilter/VariableFilter.h"
#include "marco/Diagnostic/Location.h"
#include "marco/Parser/Lexer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace marco::vf
{
  class Parser
  {
    public:
      Parser(VariableFilter& vf, std::shared_ptr<SourceFile> file);

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

    private:
      VariableFilter* vf;
      Lexer<LexerStateMachine> lexer;
      Token current;
  };
}

#endif // MARCO_VARIABLEFILTER_PARSER_H
