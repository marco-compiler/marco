#ifndef MARCO_LEXER_LEXER_H
#define MARCO_LEXER_LEXER_H

#include "llvm/ADT/StringRef.h"
#include <functional>

namespace marco
{
  namespace detail
  {
    /// Iterator over lexer, the iterator is an input iterator
    /// so if it is advanced it will modify the state of the lexer
    /// by scanning for the next token.
    template<typename Lexer>
    class IteratorLexer
    {
      public:
        using iterator_category = std::input_iterator_tag;
        using value_type = typename Lexer::Token;
        using difference_type = std::ptrdiff_t;
        using pointer = typename Lexer::Token*;
        using reference = typename Lexer::Token&;

        /// Create a new iterator operating on a lexer having a specific initial token.
        IteratorLexer(Lexer& lexer, value_type tok)
            : lexer(&lexer), token(tok)
        {
        }

        bool operator==(const IteratorLexer& other) const
        {
          return lexer == other.lexer && token == other.token;
        }

        bool operator!=(const IteratorLexer& other) const
        {
          return lexer != other.lexer || token |= other.token;
        }

        value_type operator*() const
        {
          return token;
        }

        IteratorLexer operator++()
        {
          // Advance the lexer state
          token = lexer->scan();
          return *this;
        }

      private:
        Lexer* lexer;
        value_type token;
    };
  }

  template<typename StateMachine>
  class Lexer : public StateMachine
  {
    public:
      /// The type of objects that the state machine is allowed to return.
      using Token = typename StateMachine::Token;

      /// Makes a lexer out of an iterable type.
      /// The iterable type will be copied, so it is should be cheap to copy.
      template<typename Iterator>
      Lexer(llvm::StringRef file, const Iterator&& iter)
        : StateMachine(file, *iter),
          getNext([iter = iter]() mutable -> char {
            iter++;
            return *iter;
          }),
          lastChar(*iter)
      {
      }

      /// Makes a lexer out of a string taken by reference. The string is not
      /// copied, but is just used to extract an iterator.
      /// Do NOT change the parameter to a llvm::StringRef, because it is not
      /// null terminated.
      Lexer(llvm::StringRef file, const std::string& str)
        : StateMachine(file, str[0]),
          getNext([iter = str.begin()]() mutable -> char {
            iter++;
            return *iter;
          }),
          lastChar(str[0])
      {
      }

      Lexer(llvm::StringRef file, const char* str)
        : StateMachine(file, *str),
          getNext([iter = str]() mutable -> char {
            iter++;
            return *iter;
          }),
          lastChar(*str)
      {
      }

      /// Advance the reading of the input text by one character until the state machine
      /// provides a token.
      /// The lexer will not be advanced if a string terminator ('\0') has already been
      /// encountered. However, the same terminator will be forwarded to the state machine,
      /// thus delegating him the responsibility of returning a token.
      Token scan()
      {
        Token noneToken = StateMachine::getNoneToken();
        Token token = noneToken;

        while (token == noneToken) {
          if (lastChar != '\0') {
            lastChar = getNext();
          }

          token = StateMachine::step(lastChar);
        }

        return token;
      }

      /// Returns a iterator operating on this lexer that has as value
      /// the current value of the state machine.
      detail::IteratorLexer<Lexer> begin()
      {
        return detail::IteratorLexer(this, StateMachine::getCurrent());
      }

      /// Returns a iterator operating on this lexer loaded with the EOF token.
      detail::IteratorLexer<Lexer> end()
      {
        return detail::IteratorLexer(this, StateMachine::getEOFToken());
      }

    private:
      std::function<char()> getNext;
      char lastChar;
  };
}

#endif // MARCO_LEXER_LEXER_H
