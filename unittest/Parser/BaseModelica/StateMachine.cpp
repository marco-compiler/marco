#include "marco/Parser/BaseModelica/StateMachine.h"
#include "marco/Lexer/Lexer.h"
#include "gtest/gtest.h"

using namespace ::marco;
using namespace ::marco::lexer;
using namespace ::marco::parser::bmodelica;

TEST(BaseModelica_StateMachine, defaults) {
  auto str = R"(test)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_EQ(lexer.getInt(), 0);
  EXPECT_EQ(lexer.getFloat(), 0.0);
  EXPECT_EQ(lexer.getIdentifier(), "");
  EXPECT_EQ(lexer.getString(), "");
}

TEST(BaseModelica_StateMachine, singleLineCommentsAreIgnored) {
  auto str = R"(// comment)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
  EXPECT_EQ(lexer.getCurrentPosition().line, 1);
  EXPECT_EQ(lexer.getCurrentPosition().column, 11);
}

TEST(BaseModelica_StateMachine, multiLineCommentsAreIgnored) {
  auto str =
      R"(/* comment

*/)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
  EXPECT_EQ(lexer.getCurrentPosition().line, 3);
  EXPECT_EQ(lexer.getCurrentPosition().column, 3);
}

TEST(BaseModelica_StateMachine, singleDigitInteger) {
  auto str = R"(2)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Integer>());
  EXPECT_EQ(lexer.getInt(), 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, integerValue) {
  auto str = R"(012345)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Integer>());
  EXPECT_EQ(lexer.getInt(), 12345);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, multipleIntegerValues) {
  auto str = R"(1234 5678)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

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

TEST(BaseModelica_StateMachine, floatValue) {
  auto str = R"(1.23)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::FloatingPoint>());
  EXPECT_DOUBLE_EQ(lexer.getFloat(), 1.23);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, multipleFloatValues) {
  auto str = R"(1.23 4.56)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

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

TEST(BaseModelica_StateMachine, floatsInExponentialFormat) {
  auto str = R"(2E4 3.0e-2)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

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

TEST(BaseModelica_StateMachine, floatWithDotOnly) {
  auto str = R"(2.)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::FloatingPoint>());
  EXPECT_DOUBLE_EQ(lexer.getFloat(), 2);

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, exponentialFloatWithSignOnly) {
  auto str = R"(2E-)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Error>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, exponentialFloatWithoutExponent) {
  auto str = R"(2E)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Error>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, string) {
  auto str = R"("string")";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::String>());
  EXPECT_EQ(lexer.getString(), "string");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, specialCharactersInsideString) {
  auto str = R"("\"\n\r\t\v\?")";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::String>());
  EXPECT_EQ(lexer.getString(), "\"\n\r\t\v?");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 14);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, malformedString) {
  auto str = R"(")";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Error>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, identifier) {
  auto str = R"(identifier)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Identifier>());
  EXPECT_EQ(lexer.getIdentifier(), "identifier");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 10);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, singleCharIdentifier) {
  auto str = R"(x)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Identifier>());
  EXPECT_EQ(lexer.getIdentifier(), "x");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, qIdentifier) {
  auto str = R"('identifier')";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Identifier>());
  EXPECT_EQ(lexer.getIdentifier(), "identifier");

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 12);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, algorithmKeyword) {
  auto str = R"(algorithm)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Algorithm>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, andKeyword) {
  auto str = R"(and)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::And>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, annotationKeyword) {
  auto str = R"(annotation)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Annotation>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 10);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, blockKeyword) {
  auto str = R"(block)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Block>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, breakKeyword) {
  auto str = R"(break)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Break>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, classKeyword) {
  auto str = R"(class)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Class>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, connectKeyword) {
  auto str = R"(connect)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Connect>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 7);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, connectorKeyword) {
  auto str = R"(connector)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Connector>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, constantKeyword) {
  auto str = R"(constant)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Constant>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, constrainedByKeyword) {
  auto str = R"(constrainedby)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::ConstrainedBy>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 13);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, derKeyword) {
  auto str = R"(der)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Der>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, discreteKeyword) {
  auto str = R"(discrete)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Discrete>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, eachKeyword) {
  auto str = R"(each)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Each>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, elseKeyword) {
  auto str = R"(else)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Else>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, elseIfKeyword) {
  auto str = R"(elseif)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::ElseIf>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, elseWhenKeyword) {
  auto str = R"(elsewhen)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::ElseWhen>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, encapsulatedKeyword) {
  auto str = R"(encapsulated)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Encapsulated>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 12);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, endKeyword) {
  auto str = R"(end)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::End>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, enumerationKeyword) {
  auto str = R"(enumeration)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Enumeration>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 11);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, equationKeyword) {
  auto str = R"(equation)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Equation>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, expandableKeyword) {
  auto str = R"(expandable)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Expandable>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 10);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, extendsKeyword) {
  auto str = R"(extends)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Extends>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 7);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, externalKeyword) {
  auto str = R"(external)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::External>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, falseKeyword) {
  auto str = R"(false)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::False>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, finalKeyword) {
  auto str = R"(final)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Final>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, flowKeyword) {
  auto str = R"(flow)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Flow>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, forKeyword) {
  auto str = R"(for)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::For>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, functionKeyword) {
  auto str = R"(function)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Function>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, ifKeyword) {
  auto str = R"(if)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::If>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, importKeyword) {
  auto str = R"(import)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Import>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, impureKeyword) {
  auto str = R"(impure)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Impure>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, inKeyword) {
  auto str = R"(in)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::In>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, initialKeyword) {
  auto str = R"(initial)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Initial>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 7);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, innerKeyword) {
  auto str = R"(inner)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Inner>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, inputKeyword) {
  auto str = R"(input)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Input>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, loopKeyword) {
  auto str = R"(loop)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Loop>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, modelKeyword) {
  auto str = R"(model)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Model>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, notKeyword) {
  auto str = R"(not)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Not>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 3);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, operatorKeyword) {
  auto str = R"(operator)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Operator>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 8);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, oorKeyword) {
  auto str = R"(or)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Or>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, outerKeyword) {
  auto str = R"(outer)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Outer>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, outputKeyword) {
  auto str = R"(output)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Output>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, packageKeyword) {
  auto str = R"(package)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Package>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 7);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, parameterKeyword) {
  auto str = R"(parameter)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Parameter>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, partialKeyword) {
  auto str = R"(partial)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Partial>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 7);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, protectedKeyword) {
  auto str = R"(protected)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Protected>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, publicKeyword) {
  auto str = R"(public)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Public>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, pureKeyword) {
  auto str = R"(pure)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Pure>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, recordKeyword) {
  auto str = R"(record)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Record>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, redeclareKeyword) {
  auto str = R"(redeclare)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Redeclare>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 9);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, replaceableKeyword) {
  auto str = R"(replaceable)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Replaceable>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 11);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, returnKeyword) {
  auto str = R"(return)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Return>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, streamKeyword) {
  auto str = R"(stream)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Stream>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, thenKeyword) {
  auto str = R"(then)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Then>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, trueKeyword) {
  auto str = R"(true)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::True>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, typeKeyword) {
  auto str = R"(type)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Type>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, whenKeyword) {
  auto str = R"(when)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::When>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 4);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, whileKeyword) {
  auto str = R"(while)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::While>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 5);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, withinKeyword) {
  auto str = R"(within)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Within>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 6);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, plus) {
  auto str = R"(+)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Plus>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, plusEW) {
  auto str = R"(.+)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::PlusEW>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, minus) {
  auto str = R"(-)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Minus>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, minusEW) {
  auto str = R"(.-)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::MinusEW>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, product) {
  auto str = R"(*)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Product>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, productEW) {
  auto str = R"(.*)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::ProductEW>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, division) {
  auto str = R"(/)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Division>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, divisionEW) {
  auto str = R"(./)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::DivisionEW>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, pow) {
  auto str = R"(^)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Pow>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, powEW) {
  auto str = R"(.^)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::PowEW>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, dot) {
  auto str = R"(.)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Dot>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, equal) {
  auto str = R"(==)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Equal>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, notEqual) {
  auto str = R"(<>)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::NotEqual>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, less) {
  auto str = R"(<)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Less>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, lessEqual) {
  auto str = R"(<=)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::LessEqual>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, greater) {
  auto str = R"(>)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Greater>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, greaterEqual) {
  auto str = R"(>=)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::GreaterEqual>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, comma) {
  auto str = R"(,)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Comma>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, semicolon) {
  auto str = R"(;)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Semicolon>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, colon) {
  auto str = R"(:)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::Colon>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, LPar) {
  auto str = R"(()";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::LPar>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, RPar) {
  auto str = R"())";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::RPar>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, LSquare) {
  auto str = R"([)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::LSquare>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, RSquare) {
  auto str = R"(])";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::RSquare>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, LCurly) {
  auto str = R"({)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::LCurly>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, RCurly) {
  auto str = R"(})";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::RCurly>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, equalityOperator) {
  auto str = R"(=)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EqualityOperator>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 1);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}

TEST(BaseModelica_StateMachine, assignmentOperator) {
  auto str = R"(:=)";

  auto sourceFile = std::make_shared<SourceFile>("-");
  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());
  Lexer<StateMachine> lexer(sourceFile);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::AssignmentOperator>());

  EXPECT_EQ(lexer.getTokenPosition().begin.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().begin.column, 1);

  EXPECT_EQ(lexer.getTokenPosition().end.line, 1);
  EXPECT_EQ(lexer.getTokenPosition().end.column, 2);

  EXPECT_TRUE(lexer.scan().isa<TokenKind::EndOfFile>());
}
