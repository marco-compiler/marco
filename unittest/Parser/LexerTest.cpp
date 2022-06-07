#include "gtest/gtest.h"
#include "marco/Parser/Lexer.h"
#include "marco/Parser/ModelicaStateMachine.h"

using namespace ::marco;
using namespace ::marco::parser;

TEST(ModelicaLexer, defaults)
{
  std::string str = "test";
  
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.getInt(), 0);
  EXPECT_EQ(lexer.getFloat(), 0.0);
  EXPECT_EQ(lexer.getIdentifier(), "");
  EXPECT_EQ(lexer.getString(), "");
}

TEST(ModelicaLexer, singleLineCommentsAreIgnored)
{
  std::string str = "// comment";
  
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
  EXPECT_EQ(lexer.getCurrentPosition().line, 1);
  EXPECT_EQ(lexer.getCurrentPosition().column, 11);
}

TEST(ModelicaLexer, multiLineCommentsAreIgnored)
{
  std::string str = "/* comment\n\n*/";
  
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
  EXPECT_EQ(lexer.getCurrentPosition().line, 3);
  EXPECT_EQ(lexer.getCurrentPosition().column, 3);
}

TEST(ModelicaLexer, singleDigitInteger)
{
  std::string str = "2";
  
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Integer);
  EXPECT_EQ(lexer.getInt(), 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, integerValue)
{
  std::string str = "012345";
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Integer);
  EXPECT_EQ(lexer.getInt(), 12345);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, multipleIntegerValues)
{
  std::string str = "1234 5678";
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Integer);
  EXPECT_EQ(lexer.getInt(), 1234);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_EQ(lexer.scan(), Token::Integer);
  EXPECT_EQ(lexer.getInt(), 5678);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 6);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, floatValue)
{
  std::string str = "1.23";
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::FloatingPoint);
  EXPECT_DOUBLE_EQ(lexer.getFloat(), 1.23);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, multipleFloatValues)
{
  std::string str = "1.23 4.56";
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::FloatingPoint);
  EXPECT_DOUBLE_EQ(lexer.getFloat(), 1.23);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_EQ(lexer.scan(), Token::FloatingPoint);
  EXPECT_DOUBLE_EQ(lexer.getFloat(), 4.56);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 6);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, floatsInExponentialFormat)
{
  std::string str = "2E4 3.0e-2";
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::FloatingPoint);
  EXPECT_DOUBLE_EQ(lexer.getFloat(), 20000);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_EQ(lexer.scan(), Token::FloatingPoint);
  EXPECT_DOUBLE_EQ(lexer.getFloat(), 0.03);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 5);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 10);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, floatWithDotOnly)
{
  std::string str = "2.";
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::FloatingPoint);
  EXPECT_DOUBLE_EQ(lexer.getFloat(), 2);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, exponentialFloatWithSignOnly)
{
  std::string str("2E-");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Error);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, exponentialFloatWithoutExponent)
{
  std::string str("2E");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Error);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, string)
{
  std::string str("\"string\"");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::String);
  EXPECT_EQ(lexer.getString(), "string");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, specialCharactersInsideString)
{
  std::string str("\"\\\"\\n\\r\\t\\v\\?\"");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::String);
  EXPECT_EQ(lexer.getString(), "\"\n\r\t\v?");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 14);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, malformedString)
{
  std::string str("\"");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Error);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, identifier)
{
  std::string str("identifier");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Identifier);
  EXPECT_EQ(lexer.getIdentifier(), "identifier");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 10);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, singleCharIdentifier)
{
  std::string str("x");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Identifier);
  EXPECT_EQ(lexer.getIdentifier(), "x");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, qIdentifier)
{
  std::string str("'identifier'");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Identifier);
  EXPECT_EQ(lexer.getIdentifier(), "identifier");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 12);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, qIdentifierWithEscapedChars)
{
  std::string str("'identifier\\'\\t'");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Identifier);
  EXPECT_EQ(lexer.getIdentifier(), "identifier'\t");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 16);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, algorithmKeyword)
{
  std::string str("algorithm");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Algorithm);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, andKeyword)
{
  std::string str("and");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::And);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, annotationKeyword)
{
  std::string str("annotation");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Annotation);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 10);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, blockKeyword)
{
  std::string str("block");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Block);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, breakKeyword)
{
  std::string str("break");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Break);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, classKeyword)
{
  std::string str("class");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Class);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, connectKeyword)
{
  std::string str("connect");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Connect);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 7);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, connectorKeyword)
{
  std::string str("connector");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Connector);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, constantKeyword)
{
  std::string str("constant");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Constant);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, constrainedByKeyword)
{
  std::string str("constrainedby");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::ConstrainedBy);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 13);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, derKeyword)
{
  std::string str("der");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Der);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, discreteKeyword)
{
  std::string str("discrete");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Discrete);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, eachKeyword)
{
  std::string str("each");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Each);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, elseKeyword)
{
  std::string str("else");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Else);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, elseIfKeyword)
{
  std::string str("elseif");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::ElseIf);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, elseWhenKeyword)
{
  std::string str("elsewhen");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::ElseWhen);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, encapsulatedKeyword)
{
  std::string str("encapsulated");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Encapsulated);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 12);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, endKeyword)
{
  std::string str("end");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::End);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, enumerationKeyword)
{
  std::string str("enumeration");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Enumeration);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 11);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, equationKeyword)
{
  std::string str("equation");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Equation);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, expandableKeyword)
{
  std::string str("expandable");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Expandable);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 10);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, extendsKeyword)
{
  std::string str("extends");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Extends);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 7);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, externalKeyword)
{
  std::string str("external");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::External);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, falseKeyword)
{
  std::string str("false");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::False);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, finalKeyword)
{
  std::string str("final");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Final);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, flowKeyword)
{
  std::string str("flow");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Flow);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, forKeyword)
{
  std::string str("for");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::For);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, functionKeyword)
{
  std::string str("function");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Function);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, ifKeyword)
{
  std::string str("if");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::If);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, importKeyword)
{
  std::string str("import");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Import);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, impureKeyword)
{
  std::string str("impure");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Impure);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, inKeyword)
{
  std::string str("in");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::In);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, initialKeyword)
{
  std::string str("initial");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Initial);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 7);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, innerKeyword)
{
  std::string str("inner");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Inner);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, inputKeyword)
{
  std::string str("input");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Input);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, loopKeyword)
{
  std::string str("loop");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Loop);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, modelKeyword)
{
  std::string str("model");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Model);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, notKeyword)
{
  std::string str("not");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Not);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, operatorKeyword)
{
  std::string str("operator");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Operator);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, oorKeyword)
{
  std::string str("or");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Or);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, outerKeyword)
{
  std::string str("outer");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Outer);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, outputKeyword)
{
  std::string str("output");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Output);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, packageKeyword)
{
  std::string str("package");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Package);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 7);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, parameterKeyword)
{
  std::string str("parameter");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Parameter);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, partialKeyword)
{
  std::string str("partial");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Partial);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 7);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, protectedKeyword)
{
  std::string str("protected");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Protected);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, publicKeyword)
{
  std::string str("public");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Public);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, pureKeyword)
{
  std::string str("pure");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Pure);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, recordKeyword)
{
  std::string str("record");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Record);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, redeclareKeyword)
{
  std::string str("redeclare");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Redeclare);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, replaceableKeyword)
{
  std::string str("replaceable");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Replaceable);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 11);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, returnKeyword)
{
  std::string str("return");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Return);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, streamKeyword)
{
  std::string str("stream");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Stream);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, thenKeyword)
{
  std::string str("then");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Then);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, trueKeyword)
{
  std::string str("true");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::True);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, typeKeyword)
{
  std::string str("type");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Type);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, whenKeyword)
{
  std::string str("when");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::When);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, whileKeyword)
{
  std::string str("while");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::While);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, withinKeyword)
{
  std::string str("within");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Within);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, plus)
{
  std::string str("+");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Plus);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, plusEW)
{
  std::string str(".+");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::PlusEW);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, minus)
{
  std::string str("-");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Minus);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, minusEW)
{
  std::string str(".-");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::MinusEW);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, product)
{
  std::string str("*");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Product);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, productEW)
{
  std::string str(".*");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::ProductEW);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, division)
{
  std::string str("/");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Division);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, divisionEW)
{
  std::string str("./");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::DivisionEW);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, pow)
{
  std::string str("^");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Pow);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, powEW)
{
  std::string str(".^");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::PowEW);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, dot)
{
  std::string str(".");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Dot);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, equal)
{
  std::string str("==");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Equal);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, notEqual)
{
  std::string str("<>");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::NotEqual);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, less)
{
  std::string str("<");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Less);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, lessEqual)
{
  std::string str("<=");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::LessEqual);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, greater)
{
  std::string str(">");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Greater);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, greaterEqual)
{
  std::string str(">=");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::GreaterEqual);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, comma)
{
  std::string str(",");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Comma);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, semicolon)
{
  std::string str(";");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Semicolon);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, colon)
{
  std::string str(":");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::Colon);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, LPar)
{
  std::string str("(");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::LPar);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, RPar)
{
  std::string str(")");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::RPar);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, LSquare)
{
  std::string str("[");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::LSquare);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, RSquare)
{
  std::string str("]");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::RSquare);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, LCurly)
{
  std::string str("{");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::LCurly);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, RCurly)
{
  std::string str("}");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::RCurly);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, equalityOperator)
{
  std::string str("=");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::EqualityOperator);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}

TEST(ModelicaLexer, assignmentOperator)
{
  std::string str(":=");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.scan(), Token::AssignmentOperator);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_EQ(lexer.scan(), Token::EndOfFile);
}
