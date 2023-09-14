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

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
  EXPECT_EQ(lexer.getCurrentPosition().line, 1);
  EXPECT_EQ(lexer.getCurrentPosition().column, 11);
}

TEST(ModelicaLexer, multiLineCommentsAreIgnored)
{
  std::string str = "/* comment\n\n*/";
  
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
  EXPECT_EQ(lexer.getCurrentPosition().line, 3);
  EXPECT_EQ(lexer.getCurrentPosition().column, 3);
}

TEST(ModelicaLexer, singleDigitInteger)
{
  std::string str = "2";
  
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Integer>());
  EXPECT_EQ(lexer.getInt(), 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, integerValue)
{
  std::string str = "012345";
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Integer>());
  EXPECT_EQ(lexer.getInt(), 12345);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, multipleIntegerValues)
{
  std::string str = "1234 5678";
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Integer>());
  EXPECT_EQ(lexer.getInt(), 1234);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Integer>());
  EXPECT_EQ(lexer.getInt(), 5678);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 6);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, floatValue)
{
  std::string str = "1.23";
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::FloatingPoint>());
  EXPECT_DOUBLE_EQ(lexer.getFloat(), 1.23);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, multipleFloatValues)
{
  std::string str = "1.23 4.56";
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::FloatingPoint>());
  EXPECT_DOUBLE_EQ(lexer.getFloat(), 1.23);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::FloatingPoint>());
  EXPECT_DOUBLE_EQ(lexer.getFloat(), 4.56);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 6);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, floatsInExponentialFormat)
{
  std::string str = "2E4 3.0e-2";
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::FloatingPoint>());
  EXPECT_DOUBLE_EQ(lexer.getFloat(), 20000);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::FloatingPoint>());
  EXPECT_DOUBLE_EQ(lexer.getFloat(), 0.03);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 5);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 10);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, floatWithDotOnly)
{
  std::string str = "2.";
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::FloatingPoint>());
  EXPECT_DOUBLE_EQ(lexer.getFloat(), 2);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, exponentialFloatWithSignOnly)
{
  std::string str("2E-");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Error>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, exponentialFloatWithoutExponent)
{
  std::string str("2E");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Error>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, string)
{
  std::string str("\"string\"");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::String>());
  EXPECT_EQ(lexer.getString(), "string");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, specialCharactersInsideString)
{
  std::string str("\"\\\"\\n\\r\\t\\v\\?\"");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::String>());
  EXPECT_EQ(lexer.getString(), "\"\n\r\t\v?");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 14);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, malformedString)
{
  std::string str("\"");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Error>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, identifier)
{
  std::string str("identifier");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Identifier>());
  EXPECT_EQ(lexer.getIdentifier(), "identifier");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 10);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, singleCharIdentifier)
{
  std::string str("x");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Identifier>());
  EXPECT_EQ(lexer.getIdentifier(), "x");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, qIdentifier)
{
  std::string str("'identifier'");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Identifier>());
  EXPECT_EQ(lexer.getIdentifier(), "identifier");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 12);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, qIdentifierWithEscapedChars)
{
  std::string str("'identifier\\'\\t'");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Identifier>());
  EXPECT_EQ(lexer.getIdentifier(), "identifier'\t");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 16);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, algorithmKeyword)
{
  std::string str("algorithm");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Algorithm>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, andKeyword)
{
  std::string str("and");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::And>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, annotationKeyword)
{
  std::string str("annotation");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Annotation>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 10);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, blockKeyword)
{
  std::string str("block");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Block>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, breakKeyword)
{
  std::string str("break");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Break>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, classKeyword)
{
  std::string str("class");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Class>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, connectKeyword)
{
  std::string str("connect");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Connect>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 7);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, connectorKeyword)
{
  std::string str("connector");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Connector>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, constantKeyword)
{
  std::string str("constant");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Constant>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, constrainedByKeyword)
{
  std::string str("constrainedby");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::ConstrainedBy>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 13);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, derKeyword)
{
  std::string str("der");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Der>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, discreteKeyword)
{
  std::string str("discrete");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Discrete>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, eachKeyword)
{
  std::string str("each");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Each>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, elseKeyword)
{
  std::string str("else");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Else>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, elseIfKeyword)
{
  std::string str("elseif");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::ElseIf>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, elseWhenKeyword)
{
  std::string str("elsewhen");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::ElseWhen>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, encapsulatedKeyword)
{
  std::string str("encapsulated");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Encapsulated>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 12);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, endKeyword)
{
  std::string str("end");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::End>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, enumerationKeyword)
{
  std::string str("enumeration");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Enumeration>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 11);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, equationKeyword)
{
  std::string str("equation");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Equation>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, expandableKeyword)
{
  std::string str("expandable");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Expandable>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 10);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, extendsKeyword)
{
  std::string str("extends");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Extends>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 7);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, externalKeyword)
{
  std::string str("external");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::External>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, falseKeyword)
{
  std::string str("false");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::False>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, finalKeyword)
{
  std::string str("final");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Final>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, flowKeyword)
{
  std::string str("flow");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Flow>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, forKeyword)
{
  std::string str("for");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::For>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, functionKeyword)
{
  std::string str("function");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Function>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, ifKeyword)
{
  std::string str("if");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::If>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, importKeyword)
{
  std::string str("import");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Import>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, impureKeyword)
{
  std::string str("impure");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Impure>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, inKeyword)
{
  std::string str("in");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::In>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, initialKeyword)
{
  std::string str("initial");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Initial>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 7);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, innerKeyword)
{
  std::string str("inner");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Inner>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, inputKeyword)
{
  std::string str("input");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Input>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, loopKeyword)
{
  std::string str("loop");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Loop>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, modelKeyword)
{
  std::string str("model");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Model>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, notKeyword)
{
  std::string str("not");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Not>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, operatorKeyword)
{
  std::string str("operator");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Operator>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, oorKeyword)
{
  std::string str("or");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Or>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, outerKeyword)
{
  std::string str("outer");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Outer>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, outputKeyword)
{
  std::string str("output");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Output>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, packageKeyword)
{
  std::string str("package");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Package>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 7);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, parameterKeyword)
{
  std::string str("parameter");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Parameter>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, partialKeyword)
{
  std::string str("partial");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Partial>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 7);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, protectedKeyword)
{
  std::string str("protected");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Protected>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, publicKeyword)
{
  std::string str("public");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Public>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, pureKeyword)
{
  std::string str("pure");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Pure>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, recordKeyword)
{
  std::string str("record");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Record>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, redeclareKeyword)
{
  std::string str("redeclare");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Redeclare>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, replaceableKeyword)
{
  std::string str("replaceable");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Replaceable>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 11);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, returnKeyword)
{
  std::string str("return");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Return>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, streamKeyword)
{
  std::string str("stream");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Stream>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, thenKeyword)
{
  std::string str("then");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Then>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, trueKeyword)
{
  std::string str("true");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::True>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, typeKeyword)
{
  std::string str("type");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Type>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, whenKeyword)
{
  std::string str("when");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::When>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, whileKeyword)
{
  std::string str("while");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::While>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, withinKeyword)
{
  std::string str("within");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Within>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, plus)
{
  std::string str("+");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Plus>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, plusEW)
{
  std::string str(".+");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::PlusEW>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, minus)
{
  std::string str("-");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Minus>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, minusEW)
{
  std::string str(".-");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::MinusEW>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, product)
{
  std::string str("*");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Product>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, productEW)
{
  std::string str(".*");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::ProductEW>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, division)
{
  std::string str("/");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Division>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, divisionEW)
{
  std::string str("./");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::DivisionEW>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, pow)
{
  std::string str("^");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Pow>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, powEW)
{
  std::string str(".^");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::PowEW>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, dot)
{
  std::string str(".");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Dot>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, equal)
{
  std::string str("==");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Equal>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, notEqual)
{
  std::string str("<>");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::NotEqual>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, less)
{
  std::string str("<");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Less>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, lessEqual)
{
  std::string str("<=");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::LessEqual>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, greater)
{
  std::string str(">");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Greater>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, greaterEqual)
{
  std::string str(">=");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::GreaterEqual>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, comma)
{
  std::string str(",");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Comma>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, semicolon)
{
  std::string str(";");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Semicolon>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, colon)
{
  std::string str(":");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Colon>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, LPar)
{
  std::string str("(");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::LPar>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, RPar)
{
  std::string str(")");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::RPar>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, LSquare)
{
  std::string str("[");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::LSquare>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, RSquare)
{
  std::string str("]");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::RSquare>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, LCurly)
{
  std::string str("{");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::LCurly>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, RCurly)
{
  std::string str("}");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::RCurly>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, equalityOperator)
{
  std::string str("=");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EqualityOperator>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(ModelicaLexer, assignmentOperator)
{
  std::string str(":=");
    
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Lexer<ModelicaStateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::AssignmentOperator>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}
