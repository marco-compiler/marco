#include "marco/Parser/Parser.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "gtest/gtest.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::parser;

std::unique_ptr<clang::DiagnosticsEngine> getDiagnosticsEngine() {
  clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts =
      new clang::DiagnosticOptions();

  return std::make_unique<clang::DiagnosticsEngine>(
      llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs>(
          new clang::DiagnosticIDs()),
      std::move(diagOpts),
      new clang::TextDiagnosticPrinter(llvm::errs(), diagOpts.get()));
}

TEST(Parser, rawValue_true) {
  auto str = R"(true)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseBoolValue();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ(node->getValue(), true);

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 4);
}

TEST(Parser, rawValue_false) {
  auto str = R"(false)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseBoolValue();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ(node->getValue(), false);

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 5);
}

TEST(Parser, rawValue_integer) {
  auto str = R"(012345)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseIntValue();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ(node->getValue(), 12345);

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 6);
}

TEST(Parser, rawValue_float) {
  auto str = R"(1.23)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseFloatValue();
  ASSERT_TRUE(node.has_value());

  EXPECT_DOUBLE_EQ(node->getValue(), 1.23);

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 4);
}

TEST(Parser, rawValue_string) {
  auto str = R"("test")";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseString();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ(node->getValue(), "test");

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 6);
}

TEST(Parser, identifier) {
  auto str = R"(x)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseIdentifier();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 1);

  EXPECT_EQ(node->getValue(), "x");
}

TEST(Parser, componentReference) {
  auto str = R"(x.y.z)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseComponentReference();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<ComponentReference>());
  ASSERT_EQ((*node)->cast<ComponentReference>()->getPathLength(), 3);

  EXPECT_EQ((*node)->cast<ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(
      (*node)->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
      0);

  EXPECT_EQ((*node)->cast<ComponentReference>()->getElement(1)->getName(), "y");
  EXPECT_EQ(
      (*node)->cast<ComponentReference>()->getElement(1)->getNumOfSubscripts(),
      0);

  EXPECT_EQ((*node)->cast<ComponentReference>()->getElement(2)->getName(), "z");
  EXPECT_EQ(
      (*node)->cast<ComponentReference>()->getElement(2)->getNumOfSubscripts(),
      0);
}

TEST(Parser, algorithm_emptyBody) {
  auto str = R"(algorithm)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseAlgorithmSection();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 9);

  ASSERT_TRUE((*node)->isa<Algorithm>());
  EXPECT_TRUE((*node)->cast<Algorithm>()->getStatements().empty());
}

TEST(Parser, model) {
  auto str =
      R"(model M "comment"
  Real x;
  Real y;
equation
  y = x;
end M;)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseClassDefinition();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 7);

  ASSERT_TRUE((*node)->isa<Model>());
  EXPECT_EQ((*node)->cast<Model>()->getName(), "M");
}

TEST(Parser, standardFunction) {
  auto str =
      R"(function foo
  input Real x;
  output Real y;
algorithm
  y := x;
end foo;)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseClassDefinition();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 12);

  ASSERT_TRUE((*node)->isa<StandardFunction>());
  auto function = (*node)->cast<StandardFunction>();

  EXPECT_EQ(function->getName(), "foo");
  EXPECT_EQ(function->getVariables().size(), 2);
  EXPECT_EQ(function->getAlgorithms().size(), 1);
}

TEST(Parser, partialDerFunction) {
  auto str = R"(function Bar = der(Foo, x, y);)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseClassDefinition();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 12);

  ASSERT_TRUE((*node)->isa<PartialDerFunction>());
  auto function = (*node)->cast<PartialDerFunction>();

  ASSERT_TRUE(function->getDerivedFunction()->isa<ComponentReference>());
  EXPECT_EQ(function->getDerivedFunction()
                ->cast<ComponentReference>()
                ->getPathLength(),
            1);
  EXPECT_EQ(function->getDerivedFunction()
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getName(),
            "Foo");
  EXPECT_EQ(function->getDerivedFunction()
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getNumOfSubscripts(),
            0);

  EXPECT_EQ(function->getIndependentVariables().size(), 2);

  ASSERT_TRUE(
      function->getIndependentVariables()[0]->isa<ComponentReference>());
  EXPECT_EQ(function->getIndependentVariables()[0]
                ->cast<ComponentReference>()
                ->getPathLength(),
            1);
  EXPECT_EQ(function->getIndependentVariables()[0]
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getName(),
            "x");
  EXPECT_EQ(function->getIndependentVariables()[0]
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getNumOfSubscripts(),
            0);

  ASSERT_TRUE(
      function->getIndependentVariables()[1]->isa<ComponentReference>());
  EXPECT_EQ(function->getIndependentVariables()[1]
                ->cast<ComponentReference>()
                ->getPathLength(),
            1);
  EXPECT_EQ(function->getIndependentVariables()[1]
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getName(),
            "y");
  EXPECT_EQ(function->getIndependentVariables()[1]
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getNumOfSubscripts(),
            0);
}

TEST(Parser, algorithm_statementsCount) {
  auto str =
      R"(algorithm
  x := 1;
  y := 2;
  z := 3;)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseAlgorithmSection();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 4);
  EXPECT_EQ((*node)->getLocation().end.column, 9);

  ASSERT_TRUE((*node)->isa<Algorithm>());
  EXPECT_EQ((*node)->cast<Algorithm>()->getStatements().size(), 3);
}

TEST(Parser, equalityEquation) {
  auto str = R"(y = x)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseEquation();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<EqualityEquation>());

  auto lhs = (*node)->cast<EqualityEquation>()->getLhsExpression();
  ASSERT_TRUE(lhs->isa<ComponentReference>());
  EXPECT_EQ(lhs->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(lhs->cast<ComponentReference>()->getElement(0)->getName(), "y");
  EXPECT_EQ(
      lhs->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  auto rhs = (*node)->cast<EqualityEquation>()->getRhsExpression();
  ASSERT_TRUE(rhs->isa<ComponentReference>());
  EXPECT_EQ(rhs->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(rhs->cast<ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(
      rhs->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);
}

TEST(Parser, statement_assignment) {
  auto str = R"(y := x)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseStatement();
  ASSERT_TRUE(node.has_value());

  EXPECT_TRUE((*node)->isa<AssignmentStatement>());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<AssignmentStatement>());
  auto statement = (*node)->cast<AssignmentStatement>();

  ASSERT_TRUE(statement->getDestinations()->isa<Tuple>());
  auto destinations = statement->getDestinations()->cast<Tuple>();
  ASSERT_EQ(destinations->size(), 1);

  ASSERT_TRUE(destinations->getExpression(0)->isa<ComponentReference>());
  EXPECT_EQ(destinations->getExpression(0)
                ->cast<ComponentReference>()
                ->getPathLength(),
            1);
  EXPECT_EQ(destinations->getExpression(0)
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getName(),
            "y");
  EXPECT_EQ(destinations->getExpression(0)
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getNumOfSubscripts(),
            0);

  ASSERT_TRUE(statement->getExpression()->isa<ComponentReference>());
  EXPECT_EQ(
      statement->getExpression()->cast<ComponentReference>()->getPathLength(),
      1);
  EXPECT_EQ(statement->getExpression()
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getName(),
            "x");
  EXPECT_EQ(statement->getExpression()
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getNumOfSubscripts(),
            0);
}

TEST(Parser, statement_assignmentWithMultipleDestinations) {
  auto str = R"((x, y) := Foo())";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseStatement();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 15);

  // Right-hand side is not tested because not important for this test
  ASSERT_TRUE((*node)->isa<AssignmentStatement>());
  auto statement = (*node)->cast<AssignmentStatement>();

  ASSERT_TRUE(statement->getDestinations()->isa<Tuple>());
  auto *destinations = statement->getDestinations()->cast<Tuple>();

  ASSERT_EQ(destinations->size(), 2);

  ASSERT_TRUE(destinations->getExpression(0)->isa<ComponentReference>());
  EXPECT_EQ(destinations->getExpression(0)
                ->cast<ComponentReference>()
                ->getPathLength(),
            1);
  EXPECT_EQ(destinations->getExpression(0)
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getName(),
            "x");
  EXPECT_EQ(destinations->getExpression(0)
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getNumOfSubscripts(),
            0);

  ASSERT_TRUE(destinations->getExpression(1)->isa<ComponentReference>());
  EXPECT_EQ(destinations->getExpression(1)
                ->cast<ComponentReference>()
                ->getPathLength(),
            1);
  EXPECT_EQ(destinations->getExpression(1)
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getName(),
            "y");
  EXPECT_EQ(destinations->getExpression(1)
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getNumOfSubscripts(),
            0);
}

TEST(Parser, statement_assignmentWithIgnoredResults) // NOLINT
{
  auto str = R"((x, , z) := Foo())";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseStatement();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 17);

  // Right-hand side is not tested because not so important for this test
  ASSERT_TRUE((*node)->isa<AssignmentStatement>());
  auto statement = (*node)->cast<AssignmentStatement>();

  ASSERT_TRUE(statement->getDestinations()->isa<Tuple>());
  auto *destinations = statement->getDestinations()->cast<Tuple>();

  ASSERT_EQ(destinations->size(), 3);

  ASSERT_TRUE(destinations->getExpression(0)->isa<ComponentReference>());
  EXPECT_EQ(destinations->getExpression(0)
                ->cast<ComponentReference>()
                ->getPathLength(),
            1);
  EXPECT_EQ(destinations->getExpression(0)
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getName(),
            "x");
  EXPECT_EQ(destinations->getExpression(0)
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getNumOfSubscripts(),
            0);

  ASSERT_TRUE(destinations->getExpression(1)->isa<ComponentReference>());
  EXPECT_TRUE(
      destinations->getExpression(1)->cast<ComponentReference>()->isDummy());

  ASSERT_TRUE(destinations->getExpression(2)->isa<ComponentReference>());
  EXPECT_EQ(destinations->getExpression(2)
                ->cast<ComponentReference>()
                ->getPathLength(),
            1);
  EXPECT_EQ(destinations->getExpression(2)
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getName(),
            "z");
  EXPECT_EQ(destinations->getExpression(2)
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getNumOfSubscripts(),
            0);
}

TEST(Parser, statement_function_call) {
  auto str = R"(assert(false))";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseStatement();
  ASSERT_TRUE(node.has_value());

  EXPECT_TRUE((*node)->isa<CallStatement>());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 13);

  ASSERT_TRUE((*node)->isa<CallStatement>());
  auto statement = (*node)->cast<CallStatement>();

  EXPECT_EQ(
      statement->getCall()->getCallee()->cast<ComponentReference>()->getName(),
      "assert");

  EXPECT_EQ(statement->getCall()->getNumOfArguments(), 1);
}

TEST(Parser, statement_if) {
  auto str =
      R"(if false then
  x := 1;
  y := 2;
end if;)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseStatement();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 4);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<IfStatement>());
  auto statement = (*node)->cast<IfStatement>();

  EXPECT_EQ(statement->getIfBlock()->size(), 2);
  EXPECT_EQ(statement->getNumOfElseIfBlocks(), 0);
  EXPECT_FALSE(statement->hasElseBlock());
}

TEST(Parser, statement_ifElse) {
  auto str =
      R"(if false then
  x := 1;
  y := 2;
else
  x := 1;
end if;)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseStatement();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 6);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<IfStatement>());
  auto statement = (*node)->cast<IfStatement>();

  EXPECT_EQ(statement->getIfBlock()->size(), 2);
  EXPECT_EQ(statement->getNumOfElseIfBlocks(), 0);

  ASSERT_TRUE(statement->hasElseBlock());
  EXPECT_EQ(statement->getElseBlock()->size(), 1);
}

TEST(Parser, statement_ifElseIfElse) {
  auto str =
      R"(if false then
  x := 1;
  y := 2;
  z := 3;
  t := 4;
elseif false then
  x := 1;
  y := 2;
  z := 3;
elseif false then
  x := 1;
  y := 2;
else
  x := 1;
end if;)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseStatement();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 15);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<IfStatement>());
  auto statement = (*node)->cast<IfStatement>();

  EXPECT_EQ(statement->getIfBlock()->size(), 4);

  ASSERT_EQ(statement->getNumOfElseIfBlocks(), 2);
  EXPECT_EQ(statement->getElseIfBlock(0)->size(), 3);
  EXPECT_EQ(statement->getElseIfBlock(1)->size(), 2);

  ASSERT_TRUE(statement->hasElseBlock());
  EXPECT_EQ(statement->getElseBlock()->size(), 1);
}

TEST(Parser, statement_for) {
  auto str =
      R"(for i in 1:10 loop
  x := 1;
  y := 2;
end for;)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseStatement();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 4);
  EXPECT_EQ((*node)->getLocation().end.column, 7);

  ASSERT_TRUE((*node)->isa<ForStatement>());
  auto statement = (*node)->cast<ForStatement>();

  EXPECT_EQ(statement->getStatements().size(), 2);
}

TEST(Parser, statement_while) {
  auto str =
      R"(while false loop
  x := 1;
  y := 2;
end while;)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseStatement();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 4);
  EXPECT_EQ((*node)->getLocation().end.column, 9);

  ASSERT_TRUE((*node)->isa<WhileStatement>());
  auto statement = (*node)->cast<WhileStatement>();

  EXPECT_EQ(statement->getStatements().size(), 2);
}

TEST(Parser, statement_break) {
  auto str = R"(break)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseStatement();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  EXPECT_TRUE((*node)->isa<BreakStatement>());
}

TEST(Parser, statement_return) {
  auto str = R"(return)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseStatement();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  EXPECT_TRUE((*node)->isa<ReturnStatement>());
}

TEST(Parser, expression_constant) {
  auto str = R"(012345)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Constant>());
  EXPECT_EQ((*node)->cast<Constant>()->as<int64_t>(), 12345);
}

TEST(Parser, expression_not) {
  auto str = R"(not x)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::lnot);
}

TEST(Parser, expression_and) {
  auto str = R"(x and y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 7);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::land);
}

TEST(Parser, expression_or) {
  auto str = R"(x or y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::lor);
}

TEST(Parser, expression_equal) {
  auto str = R"(x == y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::equal);
}

TEST(Parser, expression_notEqual) {
  auto str = R"(x <> y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::different);
}

TEST(Parser, expression_less) {
  auto str = R"(x < y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::less);
}

TEST(Parser, expression_lessEqual) {
  auto str = R"(x <= y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::lessEqual);
}

TEST(Parser, expression_greater) {
  auto str = R"(x > y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::greater);
}

TEST(Parser, expression_greaterEqual) {
  auto str = R"(x >= y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::greaterEqual);
}

TEST(Parser, expression_addition) {
  auto str = R"(x + y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::add);
}

TEST(Parser, expression_additionElementWise) {
  auto str = R"(x .+ y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::addEW);
}

TEST(Parser, expression_subtraction) {
  auto str = R"(x - y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::subtract);
}

TEST(Parser, expression_subtractionElementWise) {
  auto str = R"(x .- y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::subtractEW);
}

TEST(Parser, expression_multiplication) {
  auto str = R"(x * y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::multiply);
}

TEST(Parser, expression_multiplicationElementWise) {
  auto str = R"(x .* y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::multiplyEW);
}

TEST(Parser, expression_division) {
  auto str = R"(x / y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::divide);
}

TEST(Parser, expression_divisionElementWise) {
  auto str = R"(x ./ y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::divideEW);
}

TEST(Parser, expression_additionAndMultiplication) {
  auto str = R"(x + y * z)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 9);

  ASSERT_TRUE((*node)->isa<Operation>());
  ASSERT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::add);

  auto *addition = (*node)->cast<Operation>();

  auto *x = addition->getArgument(0);
  ASSERT_TRUE(x->isa<ComponentReference>());
  EXPECT_EQ(x->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(x->cast<ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(x->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
            0);

  auto *multiplication = addition->getArgument(1)->cast<Operation>();
  ASSERT_EQ(multiplication->getOperationKind(), OperationKind::multiply);

  auto *y = multiplication->getArgument(0);
  ASSERT_TRUE(y->isa<ComponentReference>());
  EXPECT_EQ(y->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(y->cast<ComponentReference>()->getElement(0)->getName(), "y");
  EXPECT_EQ(y->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
            0);

  auto *z = multiplication->getArgument(1);
  ASSERT_TRUE(z->isa<ComponentReference>());
  EXPECT_EQ(z->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(z->cast<ComponentReference>()->getElement(0)->getName(), "z");
  EXPECT_EQ(z->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
            0);
}

TEST(Parser, expression_multiplicationAndAddition) {
  auto str = R"(x * y + z)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 9);

  ASSERT_TRUE((*node)->isa<Operation>());
  ASSERT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::add);

  auto *addition = (*node)->cast<Operation>();

  auto *z = addition->getArgument(1);
  ASSERT_TRUE(z->isa<ComponentReference>());
  EXPECT_EQ(z->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(z->cast<ComponentReference>()->getElement(0)->getName(), "z");
  EXPECT_EQ(z->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
            0);

  ASSERT_TRUE(addition->getArgument(0)->isa<Operation>());

  auto *multiplication = addition->getArgument(0)->cast<Operation>();
  ASSERT_EQ(multiplication->getOperationKind(), OperationKind::multiply);

  auto *x = multiplication->getArgument(0);
  ASSERT_TRUE(x->isa<ComponentReference>());
  EXPECT_EQ(x->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(x->cast<ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(x->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
            0);

  auto *y = multiplication->getArgument(1);
  ASSERT_TRUE(y->isa<ComponentReference>());
  EXPECT_EQ(y->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(y->cast<ComponentReference>()->getElement(0)->getName(), "y");
  EXPECT_EQ(y->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
            0);
}

TEST(Parser, expression_multiplicationAndDivision) {
  auto str = R"(x * y / z)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 9);

  ASSERT_TRUE((*node)->isa<Operation>());

  auto *division = (*node)->cast<Operation>();
  ASSERT_EQ(division->getOperationKind(), OperationKind::divide);

  auto *z = division->getArgument(1);
  ASSERT_TRUE(z->isa<ComponentReference>());
  EXPECT_EQ(z->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(z->cast<ComponentReference>()->getElement(0)->getName(), "z");
  EXPECT_EQ(z->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
            0);

  ASSERT_TRUE(division->getArgument(0)->isa<Operation>());

  auto *multiplication = division->getArgument(0)->cast<Operation>();
  ASSERT_EQ(multiplication->getOperationKind(), OperationKind::multiply);

  auto *x = multiplication->getArgument(0);
  ASSERT_TRUE(x->isa<ComponentReference>());
  EXPECT_EQ(x->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(x->cast<ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(x->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
            0);

  auto *y = multiplication->getArgument(1);
  ASSERT_TRUE(y->isa<ComponentReference>());
  EXPECT_EQ(y->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(y->cast<ComponentReference>()->getElement(0)->getName(), "y");
  EXPECT_EQ(y->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
            0);
}

TEST(Parser, expression_divisionAndMultiplication) {
  auto str = R"(x / y * z)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 9);

  ASSERT_TRUE((*node)->isa<Operation>());

  auto *multiplication = (*node)->cast<Operation>();
  ASSERT_EQ(multiplication->getOperationKind(), OperationKind::multiply);

  auto *z = multiplication->getArgument(1);
  ASSERT_TRUE(z->isa<ComponentReference>());
  EXPECT_EQ(z->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(z->cast<ComponentReference>()->getElement(0)->getName(), "z");
  EXPECT_EQ(z->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
            0);

  ASSERT_TRUE(multiplication->getArgument(0)->isa<Operation>());

  auto *division = multiplication->getArgument(0)->cast<Operation>();
  ASSERT_EQ(division->getOperationKind(), OperationKind::divide);

  auto *x = division->getArgument(0);
  ASSERT_TRUE(x->isa<ComponentReference>());
  EXPECT_EQ(x->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(x->cast<ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(x->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
            0);

  auto *y = division->getArgument(1);
  ASSERT_TRUE(y->isa<ComponentReference>());
  EXPECT_EQ(y->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(y->cast<ComponentReference>()->getElement(0)->getName(), "y");
  EXPECT_EQ(y->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
            0);
}

TEST(Parser, expression_arithmeticExpressionWithParentheses) {
  auto str = R"(x / (y * z))";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 11);

  ASSERT_TRUE((*node)->isa<Operation>());

  auto *division = (*node)->cast<Operation>();
  ASSERT_EQ(division->getOperationKind(), OperationKind::divide);

  auto *x = division->getArgument(0);
  ASSERT_TRUE(x->isa<ComponentReference>());
  EXPECT_EQ(x->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(x->cast<ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(x->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
            0);

  ASSERT_TRUE(division->getArgument(1)->isa<Operation>());

  auto *multiplication = division->getArgument(1)->cast<Operation>();
  ASSERT_EQ(multiplication->getOperationKind(), OperationKind::multiply);

  auto *y = multiplication->getArgument(0);
  ASSERT_TRUE(y->isa<ComponentReference>());
  EXPECT_EQ(y->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(y->cast<ComponentReference>()->getElement(0)->getName(), "y");
  EXPECT_EQ(y->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
            0);

  auto *z = multiplication->getArgument(1);
  ASSERT_TRUE(z->isa<ComponentReference>());
  EXPECT_EQ(z->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(z->cast<ComponentReference>()->getElement(0)->getName(), "z");
  EXPECT_EQ(z->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
            0);
}

TEST(Parser, expression_pow) {
  auto str = R"(x ^ y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::powerOf);
}

TEST(Parser, expression_powElementWise) {
  auto str = R"(x .^ y)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::powerOfEW);
}

TEST(Parser, expression_tuple_empty) {
  auto str = R"(())";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 2);

  ASSERT_TRUE((*node)->isa<Tuple>());
  EXPECT_EQ((*node)->cast<Tuple>()->size(), 0);
}

TEST(Parser, expression_componentReference) {
  auto str = R"(var)";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 3);

  ASSERT_TRUE((*node)->isa<ComponentReference>());
  EXPECT_EQ((*node)->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ((*node)->cast<ComponentReference>()->getElement(0)->getName(),
            "var");
  EXPECT_EQ(
      (*node)->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
      0);
}

TEST(Parser, expression_array) {
  auto str = R"({1, 2, 3, 4, 5})";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 15);

  ASSERT_TRUE((*node)->isa<ArrayConstant>());
  const auto &array = *(*node)->cast<ArrayConstant>();
  ASSERT_EQ(array.size(), 5);

  ASSERT_TRUE(array[0]->isa<Constant>());
  EXPECT_EQ(array[0]->cast<Constant>()->as<int64_t>(), 1);

  ASSERT_TRUE(array[1]->isa<Constant>());
  EXPECT_EQ(array[1]->cast<Constant>()->as<int64_t>(), 2);

  ASSERT_TRUE(array[2]->isa<Constant>());
  EXPECT_EQ(array[2]->cast<Constant>()->as<int64_t>(), 3);

  ASSERT_TRUE(array[3]->isa<Constant>());
  EXPECT_EQ(array[3]->cast<Constant>()->as<int64_t>(), 4);

  ASSERT_TRUE(array[4]->isa<Constant>());
  EXPECT_EQ(array[4]->cast<Constant>()->as<int64_t>(), 5);
}

TEST(Parser, expression_array_induction) {
  auto str = R"({666 for i in 1:3, j in 7:12, k in 8:39})";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 40);

  ASSERT_TRUE((*node)->isa<ArrayForGenerator>());
  const auto &array = *(*node)->cast<ArrayForGenerator>();
  ASSERT_EQ(array.getNumIndices(), 3);

  ASSERT_TRUE(array.getValue()->isa<Constant>());
  EXPECT_EQ(array.getValue()->cast<Constant>()->as<int64_t>(), 666);
}

TEST(Parser, expression_subscription) {
  auto str = R"(var[3,i,:])";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 10);

  ASSERT_TRUE((*node)->isa<ComponentReference>());

  EXPECT_EQ((*node)->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ((*node)->cast<ComponentReference>()->getElement(0)->getName(),
            "var");
  EXPECT_EQ(
      (*node)->cast<ComponentReference>()->getElement(0)->getNumOfSubscripts(),
      3);

  ASSERT_FALSE((*node)
                   ->cast<ComponentReference>()
                   ->getElement(0)
                   ->getSubscript(0)
                   ->isUnbounded());
  ASSERT_TRUE((*node)
                  ->cast<ComponentReference>()
                  ->getElement(0)
                  ->getSubscript(0)
                  ->getExpression()
                  ->isa<Constant>());
  EXPECT_EQ((*node)
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getSubscript(0)
                ->getExpression()
                ->cast<Constant>()
                ->as<int64_t>(),
            3);

  ASSERT_FALSE((*node)
                   ->cast<ComponentReference>()
                   ->getElement(0)
                   ->getSubscript(1)
                   ->isUnbounded());
  ASSERT_TRUE((*node)
                  ->cast<ComponentReference>()
                  ->getElement(0)
                  ->getSubscript(1)
                  ->getExpression()
                  ->isa<ComponentReference>());
  ASSERT_EQ((*node)
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getSubscript(1)
                ->getExpression()
                ->cast<ComponentReference>()
                ->getPathLength(),
            1);
  ASSERT_EQ((*node)
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getSubscript(1)
                ->getExpression()
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getName(),
            "i");
  ASSERT_EQ((*node)
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getSubscript(1)
                ->getExpression()
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getNumOfSubscripts(),
            0);

  EXPECT_TRUE((*node)
                  ->cast<ComponentReference>()
                  ->getElement(0)
                  ->getSubscript(2)
                  ->isUnbounded());
}

TEST(Parser, expression_subscriptionOfInlineArray) {
  auto str = R"({1, 2, 3, 4, 5}[i])";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 18);

  ASSERT_TRUE((*node)->isa<Operation>());
  ASSERT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::subscription);

  auto args = (*node)->cast<Operation>()->getArguments();

  EXPECT_TRUE(args[0]->isa<ArrayConstant>());

  ASSERT_FALSE(args[1]->cast<Subscript>()->isUnbounded());
  ASSERT_TRUE(
      args[1]->cast<Subscript>()->getExpression()->isa<ComponentReference>());
}

TEST(Parser, expression_subscriptionOfFunctionCall) {
  auto str = R"(foo()[i])";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 8);

  ASSERT_TRUE((*node)->isa<Operation>());
  ASSERT_EQ((*node)->cast<Operation>()->getOperationKind(),
            OperationKind::subscription);

  auto args = (*node)->cast<Operation>()->getArguments();

  EXPECT_TRUE(args[0]->isa<Call>());

  ASSERT_FALSE(args[1]->cast<Subscript>()->isUnbounded());
  EXPECT_TRUE(
      args[1]->cast<Subscript>()->getExpression()->isa<ComponentReference>());
}

TEST(Parser, expression_functionCall_noArgs) {
  auto str = R"(foo())";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Call>());
  auto call = (*node)->cast<Call>();

  ASSERT_TRUE(call->getCallee()->isa<ComponentReference>());

  auto args = call->getArguments();
  EXPECT_TRUE(args.empty());
}

TEST(Parser, expression_functionCall_unnamedArgs) {
  auto str = R"(foo(x, 1, 2))";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 12);

  ASSERT_TRUE((*node)->isa<Call>());
  auto call = (*node)->cast<Call>();

  ASSERT_TRUE(call->getCallee()->isa<ComponentReference>());
  EXPECT_EQ(call->getCallee()->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(
      call->getCallee()->cast<ComponentReference>()->getElement(0)->getName(),
      "foo");
  EXPECT_EQ(call->getCallee()
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getNumOfSubscripts(),
            0);

  auto args = call->getArguments();
  EXPECT_EQ(args.size(), 3);

  ASSERT_TRUE(args[0]->isa<ExpressionFunctionArgument>());
  ASSERT_TRUE(args[0]
                  ->cast<ExpressionFunctionArgument>()
                  ->getExpression()
                  ->isa<ComponentReference>());
  EXPECT_EQ(args[0]
                ->cast<ExpressionFunctionArgument>()
                ->getExpression()
                ->cast<ComponentReference>()
                ->getPathLength(),
            1);
  EXPECT_EQ(args[0]
                ->cast<ExpressionFunctionArgument>()
                ->getExpression()
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getName(),
            "x");
  EXPECT_EQ(args[0]
                ->cast<ExpressionFunctionArgument>()
                ->getExpression()
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getNumOfSubscripts(),
            0);

  ASSERT_TRUE(args[1]->isa<ExpressionFunctionArgument>());
  ASSERT_TRUE(args[1]
                  ->cast<ExpressionFunctionArgument>()
                  ->getExpression()
                  ->isa<Constant>());
  EXPECT_EQ(args[1]
                ->cast<ExpressionFunctionArgument>()
                ->getExpression()
                ->cast<Constant>()
                ->as<int64_t>(),
            1);

  ASSERT_TRUE(args[2]->isa<ExpressionFunctionArgument>());
  ASSERT_TRUE(args[2]
                  ->cast<ExpressionFunctionArgument>()
                  ->getExpression()
                  ->isa<Constant>());
  EXPECT_EQ(args[2]
                ->cast<ExpressionFunctionArgument>()
                ->getExpression()
                ->cast<Constant>()
                ->as<int64_t>(),
            2);
}

TEST(Parser, expression_functionCall_namedArgs) {
  auto str = R"(foo(x = k, y = 1, z = 2))";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 24);

  ASSERT_TRUE((*node)->isa<Call>());
  auto call = (*node)->cast<Call>();

  ASSERT_TRUE(call->getCallee()->isa<ComponentReference>());
  EXPECT_EQ(call->getCallee()->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(
      call->getCallee()->cast<ComponentReference>()->getElement(0)->getName(),
      "foo");
  EXPECT_EQ(call->getCallee()
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getNumOfSubscripts(),
            0);

  auto args = call->getArguments();
  EXPECT_EQ(args.size(), 3);

  ASSERT_TRUE(args[0]->isa<NamedFunctionArgument>());
  ASSERT_TRUE(args[0]
                  ->cast<NamedFunctionArgument>()
                  ->getValue()
                  ->isa<ExpressionFunctionArgument>());
  ASSERT_TRUE(args[0]
                  ->cast<NamedFunctionArgument>()
                  ->getValue()
                  ->cast<ExpressionFunctionArgument>()
                  ->getExpression()
                  ->isa<ComponentReference>());
  EXPECT_EQ(args[0]
                ->cast<NamedFunctionArgument>()
                ->getValue()
                ->cast<ExpressionFunctionArgument>()
                ->getExpression()
                ->cast<ComponentReference>()
                ->getPathLength(),
            1);
  EXPECT_EQ(args[0]
                ->cast<NamedFunctionArgument>()
                ->getValue()
                ->cast<ExpressionFunctionArgument>()
                ->getExpression()
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getName(),
            "k");
  EXPECT_EQ(args[0]
                ->cast<NamedFunctionArgument>()
                ->getValue()
                ->cast<ExpressionFunctionArgument>()
                ->getExpression()
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getNumOfSubscripts(),
            0);

  ASSERT_TRUE(args[1]->isa<NamedFunctionArgument>());
  ASSERT_TRUE(args[1]
                  ->cast<NamedFunctionArgument>()
                  ->getValue()
                  ->isa<ExpressionFunctionArgument>());
  ASSERT_TRUE(args[1]
                  ->cast<NamedFunctionArgument>()
                  ->getValue()
                  ->cast<ExpressionFunctionArgument>()
                  ->getExpression()
                  ->isa<Constant>());
  EXPECT_EQ(args[1]
                ->cast<NamedFunctionArgument>()
                ->getValue()
                ->cast<ExpressionFunctionArgument>()
                ->getExpression()
                ->cast<Constant>()
                ->as<int64_t>(),
            1);

  ASSERT_TRUE(args[2]->isa<NamedFunctionArgument>());
  ASSERT_TRUE(args[2]
                  ->cast<NamedFunctionArgument>()
                  ->getValue()
                  ->isa<ExpressionFunctionArgument>());
  ASSERT_TRUE(args[2]
                  ->cast<NamedFunctionArgument>()
                  ->getValue()
                  ->cast<ExpressionFunctionArgument>()
                  ->getExpression()
                  ->isa<Constant>());
  EXPECT_EQ(args[2]
                ->cast<NamedFunctionArgument>()
                ->getValue()
                ->cast<ExpressionFunctionArgument>()
                ->getExpression()
                ->cast<Constant>()
                ->as<int64_t>(),
            2);
}

TEST(Parser, expression_functionCall_reductionArg) {
  auto str = R"(foo(x[i,j] for i in 1:3, j in 2:4))";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 34);

  ASSERT_TRUE((*node)->isa<Call>());
  auto call = (*node)->cast<Call>();

  ASSERT_TRUE(call->getCallee()->isa<ComponentReference>());
  EXPECT_EQ(call->getCallee()->cast<ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(
      call->getCallee()->cast<ComponentReference>()->getElement(0)->getName(),
      "foo");
  EXPECT_EQ(call->getCallee()
                ->cast<ComponentReference>()
                ->getElement(0)
                ->getNumOfSubscripts(),
            0);

  auto args = call->getArguments();
  EXPECT_EQ(args.size(), 1);

  ASSERT_TRUE(args[0]->isa<ReductionFunctionArgument>());
  ASSERT_TRUE(args[0]
                  ->cast<ReductionFunctionArgument>()
                  ->getExpression()
                  ->isa<ComponentReference>());

  ASSERT_EQ(args[0]->cast<ReductionFunctionArgument>()->getNumOfForIndices(),
            2);

  EXPECT_EQ(
      args[0]->cast<ReductionFunctionArgument>()->getForIndex(0)->getName(),
      "i");
  ASSERT_TRUE(args[0]
                  ->cast<ReductionFunctionArgument>()
                  ->getForIndex(0)
                  ->getExpression()
                  ->isa<Operation>());
  EXPECT_EQ(args[0]
                ->cast<ReductionFunctionArgument>()
                ->getForIndex(0)
                ->getExpression()
                ->cast<Operation>()
                ->getOperationKind(),
            OperationKind::range);

  EXPECT_EQ(
      args[0]->cast<ReductionFunctionArgument>()->getForIndex(1)->getName(),
      "j");
  ASSERT_TRUE(args[0]
                  ->cast<ReductionFunctionArgument>()
                  ->getForIndex(1)
                  ->getExpression()
                  ->isa<Operation>());
  EXPECT_EQ(args[0]
                ->cast<ReductionFunctionArgument>()
                ->getForIndex(1)
                ->getExpression()
                ->cast<Operation>()
                ->getOperationKind(),
            OperationKind::range);
}

TEST(Parser, annotation_inlineTrue) {
  auto str = R"(annotation(inline = true))";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseAnnotation();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 25);

  ASSERT_TRUE((*node)->isa<Annotation>());
  EXPECT_TRUE((*node)->cast<Annotation>()->getInlineProperty());
}

TEST(Parser, annotation_inlineFalse) {
  auto str = R"(annotation(inline = false))";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseAnnotation();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 26);

  ASSERT_TRUE((*node)->isa<Annotation>());
  EXPECT_FALSE((*node)->cast<Annotation>()->getInlineProperty());
}

TEST(Parser, annotation_inverseFunction) {
  auto str = R"(annotation(inverse(y = foo1(x, z), z = foo2(x, y))))";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseAnnotation();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 51);

  ASSERT_TRUE((*node)->isa<Annotation>());
  auto annotation = (*node)->cast<Annotation>()->getInverseFunctionAnnotation();

  ASSERT_TRUE(annotation.isInvertible("y"));
  EXPECT_EQ(annotation.getInverseFunction("y"), "foo1");
  ASSERT_EQ(annotation.getInverseArgs("y").size(), 2);
  EXPECT_EQ(annotation.getInverseArgs("y")[0], "x");
  EXPECT_EQ(annotation.getInverseArgs("y")[1], "z");

  ASSERT_TRUE(annotation.isInvertible("z"));
  EXPECT_EQ(annotation.getInverseFunction("z"), "foo2");
  ASSERT_EQ(annotation.getInverseArgs("z").size(), 2);
  EXPECT_EQ(annotation.getInverseArgs("z")[0], "x");
  EXPECT_EQ(annotation.getInverseArgs("z")[1], "y");
}

TEST(Parser, annotation_functionDerivativeWithOrder) {
  auto str = R"(annotation(derivative(order=2)=foo1))";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseAnnotation();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 36);

  ASSERT_TRUE((*node)->isa<Annotation>());
  auto annotation = (*node)->cast<Annotation>()->getDerivativeAnnotation();

  EXPECT_EQ(annotation.getName(), "foo1");
  EXPECT_EQ(annotation.getOrder(), 2);
}

TEST(Parser, annotation_functionDerivativeWithoutOrder) {
  auto str = R"(annotation(derivative=foo1))";

  auto sourceFile = std::make_shared<SourceFile>("test.mo");

  auto diagnostics = getDiagnosticsEngine();
  clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
  auto &sourceManager = fileSourceMgr.get();

  auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
  sourceFile->setMemoryBuffer(buffer.get());

  Parser parser(*diagnostics, sourceManager, sourceFile);

  auto node = parser.parseAnnotation();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 27);

  ASSERT_TRUE((*node)->isa<Annotation>());
  auto annotation = (*node)->cast<Annotation>()->getDerivativeAnnotation();

  EXPECT_EQ(annotation.getName(), "foo1");
  EXPECT_EQ(annotation.getOrder(), 1);
}
