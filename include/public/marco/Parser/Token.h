#ifndef MARCO_PARSER_TOKEN_H
#define MARCO_PARSER_TOKEN_H

#include <iostream>
#include <string>

namespace llvm
{
  class raw_ostream;
}

namespace marco::parser
{
  enum class Token
  {
    // Control tokens.
    None,
    Begin,
    EndOfFile,
    Error,

    // Placeholders.
    // The actual values are contained within the lexer.
    Identifier,
    Integer,
    FloatingPoint,
    String,

    // Reserved language keywords.
    Algorithm,
    And,
    Annotation,
    Block,
    Break,
    Class,
    Connect,
    Connector,
    Constant,
    ConstrainedBy,
    Der,
    Discrete,
    Each,
    Else,
    ElseIf,
    ElseWhen,
    Encapsulated,
    End,
    Enumeration,
    Equation,
    Expandable,
    Extends,
    External,
    False,
    Final,
    Flow,
    For,
    Function,
    If,
    Import,
    Impure,
    In,
    Initial,
    Inner,
    Input,
    Loop,
    Model,
    Not,
    Operator,
    Or,
    Outer,
    Output,
    Package,
    Parameter,
    Partial,
    Protected,
    Public,
    Pure,
    Record,
    Redeclare,
    Replaceable,
    Return,
    Stream,
    Then,
    True,
    Type,
    When,
    While,
    Within,

    // Symbols.
    Plus,
    PlusEW,
    Minus,
    MinusEW,
    Product,
    ProductEW,
    Division,
    DivisionEW,
    Pow,
    PowEW,
    Dot,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Comma,
    Semicolon,
    Colon,
    LPar,
    RPar,
    LSquare,
    RSquare,
    LCurly,
    RCurly,
    EqualityOperator,
    AssignmentOperator
  };

  std::string toString(Token obj);

  std::ostream& operator<<(std::ostream& os, const Token& obj);

  llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Token& obj);
}

#endif // MARCO_PARSER_TOKEN_H
