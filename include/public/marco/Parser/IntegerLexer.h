#ifndef MARCO_PARSER_INTEGERLEXER_H
#define MARCO_PARSER_INTEGERLEXER_H

#include <cstdint>

namespace marco {
/// An integer lexer is a object that can be feed with
/// integers a1, a2 ... aN and will build the integer
/// (a1*base^N) + (a2*base^N-1) ... + (aN*base^1);
///
/// It is used to parse unsigned integers.
template <unsigned int Base = 10>
class IntegerLexer {
public:
  /// Creates a new IntegerLexer loaded with the provided int.
  IntegerLexer(int64_t value = 0) : value(value) {}

  /// Adds the next element to the integer.
  IntegerLexer operator+=(int i) {
    value = value * Base;
    value = value + i;
    return *this;
  }

  /// Returns the currently built value.
  int64_t get() const { return value; }

  void reset() { value = 0; }

private:
  int64_t value;
};
} // namespace marco

#endif // MARCO_PARSER_INTEGERLEXER_H
