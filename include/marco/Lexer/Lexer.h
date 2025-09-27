#ifndef MARCO_LEXER_LEXER_H
#define MARCO_LEXER_LEXER_H

#include "marco/Lexer/Location.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <memory>

namespace marco::lexer {
namespace detail {
/// Iterator over lexer, the iterator is an input iterator
/// so if it is advanced it will modify the state of the lexer
/// by scanning for the next token.
template <typename Lexer>
class IteratorLexer {
public:
  using iterator_category = std::input_iterator_tag;
  using value_type = typename Lexer::Token;
  using difference_type = std::ptrdiff_t;
  using pointer = typename Lexer::Token *;
  using reference = typename Lexer::Token &;

  /// Create a new iterator operating on a lexer having a specific initial
  /// token.
  IteratorLexer(Lexer &lexer, value_type tok) : lexer(&lexer), token(tok) {}

  bool operator==(const IteratorLexer &other) const {
    return lexer == other.lexer && token == other.token;
  }

  bool operator!=(const IteratorLexer &other) const {
    return lexer != other.lexer || token |= other.token;
  }

  value_type operator*() const { return token; }

  IteratorLexer operator++() {
    // Advance the lexer state
    token = lexer->scan();
    return *this;
  }

private:
  Lexer *lexer;
  value_type token;
};
} // namespace detail

template <typename Token>
struct TokenTraits {
  // static Token getEOFToken();

  using Id = typename Token::UnknownTokenTypeError;
};

template <typename StateMachine>
class Lexer : public StateMachine {
public:
  /// The type of objects that the state machine is allowed to return.
  using Token = typename StateMachine::Token;
  using TokenTraits = typename lexer::TokenTraits<Token>;

  explicit Lexer(std::shared_ptr<SourceFile> file)
      : StateMachine(file, *file->getBuffer()->getBufferStart()),
        getNext([iter = file->getBuffer()->getBufferStart()]() mutable -> char {
          iter++;
          return *iter;
        }),
        lastChar(*file->getBuffer()->getBufferStart()) {}

  /// Advance the reading of the input text by one character until the state
  /// machine provides a token. The lexer will not be advanced if a string
  /// terminator ('\0') has already been encountered. However, the same
  /// terminator will be forwarded to the state machine, thus delegating him the
  /// responsibility of returning a token.
  Token scan() {
    std::optional<Token> token = std::nullopt;

    while (!token) {
      if (lastChar != '\0') {
        lastChar = getNext();
      }

      token = StateMachine::step(lastChar);
    }

    return *token;
  }

  /// Returns a iterator operating on this lexer that has as value
  /// the current value of the state machine.
  detail::IteratorLexer<Lexer> begin() {
    return detail::IteratorLexer(this, StateMachine::getCurrent());
  }

  /// Returns a iterator operating on this lexer loaded with the EOF token.
  detail::IteratorLexer<Lexer> end() {
    return detail::IteratorLexer(this, TokenTraits::getEOFToken());
  }

private:
  std::function<char()> getNext;
  char lastChar;
};
} // namespace marco::lexer

#endif // MARCO_LEXER_LEXER_H
