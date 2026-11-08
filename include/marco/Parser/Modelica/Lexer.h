#ifndef MARCO_PARSER_MODELICA_LEXER_H
#define MARCO_PARSER_MODELICA_LEXER_H

#include "marco/Parser/Modelica/Token.h"
#include <memory>
#include <vector>

namespace marco::parser::modelica {
/// Lexer of the Modelica language. Comments are skipped,
/// and malformed input is reported by 'Error' tokens whose text describes the
/// problem. The last token is always 'EndOfFile' or 'Error'.
class Lexer {
public:
  explicit Lexer(std::shared_ptr<SourceFile> file);

  /// Scan the next token.
  Token next();

  /// Scan all the tokens, up to the end of the file or the first error.
  std::vector<Token> tokenize();

private:
  char peek(size_t offset = 0) const;
  void advance();
  SourcePosition position() const;

  bool skipWhitespaceAndComments(Token &error);

  Token makeToken(TokenKind kind, SourcePosition begin) const;
  Token makeError(SourcePosition begin, llvm::StringRef message) const;

  Token scanIdentifier();
  Token scanQuotedIdentifier();
  Token scanNumber();
  Token scanString();
  Token scanSymbol();

private:
  std::shared_ptr<SourceFile> file;
  llvm::StringRef buffer;
  size_t index = 0;
  int64_t line = 1;
  int64_t column = 1;

  /// Position of the last consumed character.
  SourcePosition lastPosition;
};
} // namespace marco::parser::modelica

#endif // MARCO_PARSER_MODELICA_LEXER_H
