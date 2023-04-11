#include "gtest/gtest.h"
#include "marco/Diagnostic/Printer.h"
#include "marco/Parser/Parser.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::diagnostic;
using namespace ::marco::parser;

TEST(Parser, rawValue_true)
{
  std::string str = "true";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseBoolValue();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ(node->getValue(), true);

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 4);
}

TEST(Parser, rawValue_false)
{
  std::string str = "false";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseBoolValue();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ(node->getValue(), false);

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 5);
}

TEST(Parser, rawValue_integer)
{
  std::string str = "012345";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseIntValue();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ(node->getValue(), 12345);

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 6);
}

TEST(Parser, rawValue_float)
{
  std::string str = "1.23";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseFloatValue();
  ASSERT_TRUE(node.has_value());

  EXPECT_DOUBLE_EQ(node->getValue(), 1.23);

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 4);
}

TEST(Parser, rawValue_string)
{
  std::string str = "\"test\"";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseString();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ(node->getValue(), "test");

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 6);
}

TEST(Parser, identifier)
{
  std::string str = "x.y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseIdentifier();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 3);

  EXPECT_EQ(node->getValue(), "x.y");
}

TEST(Parser, algorithm_emptyBody)
{
  std::string str = "algorithm";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseAlgorithmSection();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 9);

  ASSERT_TRUE((*node)->isa<Algorithm>());
  EXPECT_TRUE((*node)->cast<Algorithm>()->getStatements().empty());
}

TEST(Parser, model)
{
  std::string str = "model M \"comment\"\n"
                    "  Real x;\n"
                    "  Real y;\n"
                    "equation\n"
                    "  y = x;"
                    "end M;";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseClassDefinition();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 7);

  ASSERT_TRUE((*node)->isa<Model>());
  EXPECT_EQ((*node)->cast<Model>()->getName(), "M");
}

TEST(Parser, standardFunction)
{
  std::string str = "function foo\n"
                    "  input Real x;\n"
                    "  output Real y;\n"
                    "algorithm\n"
                    "  y := x;\n"
                    "end foo;";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

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

TEST(Parser, partialDerFunction)
{
  std::string str = "function Bar = der(Foo, x, y);";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseClassDefinition();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 12);

  ASSERT_TRUE((*node)->isa<PartialDerFunction>());
  auto function = (*node)->cast<PartialDerFunction>();

  ASSERT_TRUE(function->getDerivedFunction()->isa<ast::ComponentReference>());
  EXPECT_EQ(function->getDerivedFunction()->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(function->getDerivedFunction()->cast<ast::ComponentReference>()->getElement(0)->getName(), "Foo");
  EXPECT_EQ(function->getDerivedFunction()->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  EXPECT_EQ(function->getIndependentVariables().size(), 2);

  ASSERT_TRUE(function->getIndependentVariables()[0]->isa<ast::ComponentReference>());
  EXPECT_EQ(function->getIndependentVariables()[0]->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(function->getIndependentVariables()[0]->cast<ast::ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(function->getIndependentVariables()[0]->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  ASSERT_TRUE(function->getIndependentVariables()[1]->isa<ast::ComponentReference>());
  EXPECT_EQ(function->getIndependentVariables()[1]->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(function->getIndependentVariables()[1]->cast<ast::ComponentReference>()->getElement(0)->getName(), "y");
  EXPECT_EQ(function->getIndependentVariables()[1]->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);
}

TEST(Parser, algorithm_statementsCount)
{
  std::string str = "algorithm\n"
                    "	x := 1;\n"
                    "	y := 2;\n"
                    "	z := 3;";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseAlgorithmSection();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 4);
  EXPECT_EQ((*node)->getLocation().end.column, 8);

  ASSERT_TRUE((*node)->isa<Algorithm>());
  EXPECT_EQ((*node)->cast<Algorithm>()->getStatements().size(), 3);
}

TEST(Parser, equation)
{
  std::string str = "y = x";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseEquation();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Equation>());

  auto lhs = (*node)->cast<Equation>()->getLhsExpression();
  ASSERT_TRUE(lhs->isa<ast::ComponentReference>());
  EXPECT_EQ(lhs->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(lhs->cast<ast::ComponentReference>()->getElement(0)->getName(), "y");
  EXPECT_EQ(lhs->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  auto rhs = (*node)->cast<Equation>()->getRhsExpression();
  ASSERT_TRUE(rhs->isa<ast::ComponentReference>());
  EXPECT_EQ(rhs->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(rhs->cast<ast::ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(rhs->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);
}

TEST(Parser, statement_assignment)
{
  std::string str = "y := x";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

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

  ASSERT_TRUE(destinations->getExpression(0)->isa<ast::ComponentReference>());
  EXPECT_EQ(destinations->getExpression(0)->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(destinations->getExpression(0)->cast<ast::ComponentReference>()->getElement(0)->getName(), "y");
  EXPECT_EQ(destinations->getExpression(0)->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  ASSERT_TRUE(statement->getExpression()->isa<ast::ComponentReference>());
  EXPECT_EQ(statement->getExpression()->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(statement->getExpression()->cast<ast::ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(statement->getExpression()->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);
}

TEST(Parser, statement_assignmentWithMultipleDestinations)
{
  std::string str = "(x, y) := Foo()";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

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
  auto* destinations = statement->getDestinations()->cast<Tuple>();

  ASSERT_EQ(destinations->size(), 2);

  ASSERT_TRUE(destinations->getExpression(0)->isa<ast::ComponentReference>());
  EXPECT_EQ(destinations->getExpression(0)->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(destinations->getExpression(0)->cast<ast::ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(destinations->getExpression(0)->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  ASSERT_TRUE(destinations->getExpression(1)->isa<ast::ComponentReference>());
  EXPECT_EQ(destinations->getExpression(1)->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(destinations->getExpression(1)->cast<ast::ComponentReference>()->getElement(0)->getName(), "y");
  EXPECT_EQ(destinations->getExpression(1)->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);
}

TEST(Parser, statement_assignmentWithIgnoredResults)	 // NOLINT
{
  std::string str = "(x, , z) := Foo()";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

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
  auto* destinations = statement->getDestinations()->cast<Tuple>();

  ASSERT_EQ(destinations->size(), 3);

  ASSERT_TRUE(destinations->getExpression(0)->isa<ast::ComponentReference>());
  EXPECT_EQ(destinations->getExpression(0)->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(destinations->getExpression(0)->cast<ast::ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(destinations->getExpression(0)->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  ASSERT_TRUE(destinations->getExpression(1)->isa<ast::ComponentReference>());
  EXPECT_TRUE(destinations->getExpression(1)->cast<ast::ComponentReference>()->isDummy());

  ASSERT_TRUE(destinations->getExpression(2)->isa<ast::ComponentReference>());
  EXPECT_EQ(destinations->getExpression(2)->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(destinations->getExpression(2)->cast<ast::ComponentReference>()->getElement(0)->getName(), "z");
  EXPECT_EQ(destinations->getExpression(2)->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);
}

TEST(Parser, statement_if)
{
  std::string str = "if false then\n"
                    "  x := 1;\n"
                    "  y := 2;\n"
                    "end if;";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

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

TEST(Parser, statement_ifElse)
{
  std::string str = "if false then\n"
                    "  x := 1;\n"
                    "  y := 2;\n"
                    "else\n"
                    "  x := 1;\n"
                    "end if;\n";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

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

TEST(Parser, statement_ifElseIfElse)
{
  std::string str = "if false then\n"
                    "  x := 1;\n"
                    "  y := 2;\n"
                    "  z := 3;\n"
                    "  y := 4;\n"
                    "elseif false then\n"
                    "  x := 1;\n"
                    "  y := 2;\n"
                    "  z := 3;\n"
                    "elseif false then\n"
                    "  x := 1;\n"
                    "  y := 2;\n"
                    "else\n"
                    "  x := 1;\n"
                    "end if;";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

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

TEST(Parser, statement_for)
{
  std::string str = "for i in 1:10 loop\n"
                    "  x := 1;\n"
                    "  y := 2;\n"
                    "end for;";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

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

TEST(Parser, statement_while)
{
  std::string str = "while false loop\n"
                    "  x := 1;\n"
                    "  y := 2;\n"
                    "end while;";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

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

TEST(Parser, statement_break)
{
  std::string str = "break";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseStatement();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  EXPECT_TRUE((*node)->isa<BreakStatement>());
}

TEST(Parser, statement_return)
{
  std::string str = "return";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseStatement();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  EXPECT_TRUE((*node)->isa<ReturnStatement>());
}

TEST(Parser, expression_constant)
{
  std::string str = "012345";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Constant>());
  EXPECT_EQ((*node)->cast<Constant>()->as<int64_t>(), 12345);
}

TEST(Parser, expression_not)
{
  std::string str = "not x";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::lnot);
}

TEST(Parser, expression_and)
{
  std::string str = "x and y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 7);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::land);
}

TEST(Parser, expression_or)
{
  std::string str = "x or y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::lor);
}

TEST(Parser, expression_equal)
{
  std::string str = "x == y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::equal);
}

TEST(Parser, expression_notEqual)
{
  std::string str = "x <> y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::different);
}

TEST(Parser, expression_less)
{
  std::string str = "x < y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::less);
}

TEST(Parser, expression_lessEqual)
{
  std::string str = "x <= y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::lessEqual);
}

TEST(Parser, expression_greater)
{
  std::string str = "x > y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::greater);
}

TEST(Parser, expression_greaterEqual)
{
  std::string str = "x >= y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::greaterEqual);
}

TEST(Parser, expression_addition)
{
  std::string str = "x + y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::add);
}

TEST(Parser, expression_additionElementWise)
{
  std::string str = "x .+ y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::addEW);
}

TEST(Parser, expression_subtraction)
{
  std::string str = "x - y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::subtract);
}

TEST(Parser, expression_subtractionElementWise)
{
  std::string str = "x .- y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::subtractEW);
}

TEST(Parser, expression_multiplication)
{
  std::string str = "x * y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::multiply);
}

TEST(Parser, expression_multiplicationElementWise)
{
  std::string str = "x .* y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::multiplyEW);
}

TEST(Parser, expression_division)
{
  std::string str = "x / y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::divide);
}

TEST(Parser, expression_divisionElementWise)
{
  std::string str = "x ./ y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::divideEW);
}

TEST(Parser, expression_additionAndMultiplication)
{
  std::string str = "x + y * z";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 9);

  ASSERT_TRUE((*node)->isa<Operation>());
  ASSERT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::add);

  auto* addition = (*node)->cast<Operation>();

  auto* x = addition->getArgument(0);
  ASSERT_TRUE(x->isa<ast::ComponentReference>());
  EXPECT_EQ(x->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(x->cast<ast::ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(x->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  auto* multiplication = addition->getArgument(1)->cast<Operation>();
  ASSERT_EQ(multiplication->getOperationKind(), OperationKind::multiply);

  auto* y = multiplication->getArgument(0);
  ASSERT_TRUE(y->isa<ast::ComponentReference>());
  EXPECT_EQ(y->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(y->cast<ast::ComponentReference>()->getElement(0)->getName(), "y");
  EXPECT_EQ(y->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  auto* z = multiplication->getArgument(1);
  ASSERT_TRUE(z->isa<ast::ComponentReference>());
  EXPECT_EQ(z->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(z->cast<ast::ComponentReference>()->getElement(0)->getName(), "z");
  EXPECT_EQ(z->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);
}

TEST(Parser, expression_multiplicationAndAddition)
{
  std::string str = "x * y + z";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 9);

  ASSERT_TRUE((*node)->isa<Operation>());
  ASSERT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::add);

  auto* addition = (*node)->cast<Operation>();

  auto* z = addition->getArgument(1);
  ASSERT_TRUE(z->isa<ast::ComponentReference>());
  EXPECT_EQ(z->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(z->cast<ast::ComponentReference>()->getElement(0)->getName(), "z");
  EXPECT_EQ(z->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  ASSERT_TRUE(addition->getArgument(0)->isa<Operation>());

  auto* multiplication = addition->getArgument(0)->cast<Operation>();
  ASSERT_EQ(multiplication->getOperationKind(), OperationKind::multiply);

  auto* x = multiplication->getArgument(0);
  ASSERT_TRUE(x->isa<ast::ComponentReference>());
  EXPECT_EQ(x->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(x->cast<ast::ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(x->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  auto* y = multiplication->getArgument(1);
  ASSERT_TRUE(y->isa<ast::ComponentReference>());
  EXPECT_EQ(y->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(y->cast<ast::ComponentReference>()->getElement(0)->getName(), "y");
  EXPECT_EQ(y->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);
}

TEST(Parser, expression_multiplicationAndDivision)
{
  std::string str = "x * y / z";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 9);

  ASSERT_TRUE((*node)->isa<Operation>());

  auto* division = (*node)->cast<Operation>();
  ASSERT_EQ(division->getOperationKind(), OperationKind::divide);

  auto* z = division->getArgument(1);
  ASSERT_TRUE(z->isa<ast::ComponentReference>());
  EXPECT_EQ(z->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(z->cast<ast::ComponentReference>()->getElement(0)->getName(), "z");
  EXPECT_EQ(z->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  ASSERT_TRUE(division->getArgument(0)->isa<Operation>());

  auto* multiplication = division->getArgument(0)->cast<Operation>();
  ASSERT_EQ(multiplication->getOperationKind(), OperationKind::multiply);

  auto* x = multiplication->getArgument(0);
  ASSERT_TRUE(x->isa<ast::ComponentReference>());
  EXPECT_EQ(x->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(x->cast<ast::ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(x->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  auto* y = multiplication->getArgument(1);
  ASSERT_TRUE(y->isa<ast::ComponentReference>());
  EXPECT_EQ(y->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(y->cast<ast::ComponentReference>()->getElement(0)->getName(), "xy");
  EXPECT_EQ(y->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);
}

TEST(Parser, expression_divisionAndMultiplication)
{
  std::string str = "x / y * z";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 9);

  ASSERT_TRUE((*node)->isa<Operation>());

  auto* multiplication = (*node)->cast<Operation>();
  ASSERT_EQ(multiplication->getOperationKind(), OperationKind::multiply);

  auto* z = multiplication->getArgument(1);
  ASSERT_TRUE(z->isa<ast::ComponentReference>());
  EXPECT_EQ(z->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(z->cast<ast::ComponentReference>()->getElement(0)->getName(), "z");
  EXPECT_EQ(z->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  ASSERT_TRUE(multiplication->getArgument(0)->isa<Operation>());

  auto* division = multiplication->getArgument(0)->cast<Operation>();
  ASSERT_EQ(division->getOperationKind(), OperationKind::divide);

  auto* x = division->getArgument(0);
  ASSERT_TRUE(x->isa<ast::ComponentReference>());
  EXPECT_EQ(x->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(x->cast<ast::ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(x->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  auto* y = division->getArgument(1);
  ASSERT_TRUE(y->isa<ast::ComponentReference>());
  EXPECT_EQ(y->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(y->cast<ast::ComponentReference>()->getElement(0)->getName(), "y");
  EXPECT_EQ(y->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);
}

TEST(Parser, expression_arithmeticExpressionWithParentheses)
{
  std::string str = "x / (y * z)";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 11);

  ASSERT_TRUE((*node)->isa<Operation>());

  auto* division = (*node)->cast<Operation>();
  ASSERT_EQ(division->getOperationKind(), OperationKind::divide);

  auto* x = division->getArgument(0);
  ASSERT_TRUE(x->isa<ast::ComponentReference>());
  EXPECT_EQ(x->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(x->cast<ast::ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(x->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  ASSERT_TRUE(division->getArgument(1)->isa<Operation>());

  auto* multiplication = division->getArgument(1)->cast<Operation>();
  ASSERT_EQ(multiplication->getOperationKind(), OperationKind::multiply);

  auto* y = multiplication->getArgument(0);
  ASSERT_TRUE(y->isa<ast::ComponentReference>());
  EXPECT_EQ(y->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(y->cast<ast::ComponentReference>()->getElement(0)->getName(), "y");
  EXPECT_EQ(y->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  auto* z = multiplication->getArgument(1);
  ASSERT_TRUE(z->isa<ast::ComponentReference>());
  EXPECT_EQ(z->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(z->cast<ast::ComponentReference>()->getElement(0)->getName(), "z");
  EXPECT_EQ(z->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);
}

TEST(Parser, expression_pow)
{
  std::string str = "x ^ y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::powerOf);
}

TEST(Parser, expression_powElementWise)
{
  std::string str = "x .^ y";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::powerOfEW);
}

TEST(Parser, expression_tuple_empty)
{
  std::string str = "()";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 2);

  ASSERT_TRUE((*node)->isa<Tuple>());
  EXPECT_EQ((*node)->cast<Tuple>()->size(), 0);
}

TEST(Parser, expression_componentReference)
{
  std::string str = "var";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 3);

  ASSERT_TRUE((*node)->isa<ast::ComponentReference>());
  EXPECT_EQ((*node)->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ((*node)->cast<ast::ComponentReference>()->getElement(0)->getName(), "var");
  EXPECT_EQ((*node)->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);
}

TEST(Parser, expression_array)
{
  std::string str = "{1, 2, 3, 4, 5}";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 15);

  ASSERT_TRUE((*node)->isa<Array>());
  const auto& array = *(*node)->cast<Array>();
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

TEST(Parser, expression_subscription)
{
  std::string str = "var[3,i,:]";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 10);

  ASSERT_TRUE((*node)->isa<Operation>());
  auto args =(*node)->cast<Operation>()->getArguments();

  ASSERT_TRUE((*node)->isa<ast::ComponentReference>());

  EXPECT_EQ((*node)->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ((*node)->cast<ast::ComponentReference>()->getElement(0)->getName(), "var");
  EXPECT_EQ((*node)->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 3);

  ASSERT_TRUE((*node)->cast<ast::ComponentReference>()->getElement(0)->getSubscript(0)->isa<Constant>());
  EXPECT_EQ((*node)->cast<ast::ComponentReference>()->getElement(0)->getSubscript(0)->cast<Constant>()->as<int64_t>(), 3);

  ASSERT_TRUE((*node)->cast<ast::ComponentReference>()->getElement(0)->getSubscript(2)->isa<ast::ComponentReference>());

  ASSERT_TRUE((*node)->cast<ast::ComponentReference>()->getElement(0)->getSubscript(2)->isa<Constant>());
  EXPECT_EQ((*node)->cast<ast::ComponentReference>()->getElement(0)->getSubscript(2)->cast<Constant>()->as<int64_t>(), 3);
}

TEST(Parser, expression_subscriptionOfInlineArray)
{
  std::string str = "{1, 2, 3, 4, 5}[i]";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 18);

  ASSERT_TRUE((*node)->isa<Operation>());
  ASSERT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::subscription);

  auto args = (*node)->cast<Operation>()->getArguments();

  EXPECT_TRUE(args[0]->isa<Array>());
  ASSERT_TRUE(args[1]->isa<ast::ComponentReference>());
}

TEST(Parser, expression_subscriptionOfFunctionCall)
{
  std::string str = "foo()[i]";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 8);

  ASSERT_TRUE((*node)->isa<Operation>());
  ASSERT_EQ((*node)->cast<Operation>()->getOperationKind(), OperationKind::subscription);

  auto args = (*node)->cast<Operation>()->getArguments();

  EXPECT_TRUE(args[0]->isa<Call>());
  ASSERT_TRUE(args[1]->isa<ComponentReference>());
}

TEST(Parser, expression_functionCall_noArgs)
{
  std::string str = "foo()";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

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

TEST(Parser, expression_functionCall_withArgs)
{
  std::string str = "foo(x, 1, 2)";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseExpression();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 12);

  ASSERT_TRUE((*node)->isa<Call>());
  auto call = (*node)->cast<Call>();

  ASSERT_TRUE(call->getCallee()->isa<ast::ComponentReference>());
  EXPECT_EQ(call->getCallee()->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(call->getCallee()->cast<ast::ComponentReference>()->getElement(0)->getName(), "Foo");
  EXPECT_EQ(call->getCallee()->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  auto args = call->getArguments();
  EXPECT_EQ(args.size(), 3);

  ASSERT_TRUE(args[0]->isa<ast::ComponentReference>());
  EXPECT_EQ(args[0]->cast<ast::ComponentReference>()->getPathLength(), 1);
  EXPECT_EQ(args[0]->cast<ast::ComponentReference>()->getElement(0)->getName(), "x");
  EXPECT_EQ(args[0]->cast<ast::ComponentReference>()->getElement(0)->getNumOfSubscripts(), 0);

  ASSERT_TRUE(args[1]->isa<Constant>());
  EXPECT_EQ(args[1]->cast<Constant>()->as<int64_t>(), 1);

  ASSERT_TRUE(args[2]->isa<Constant>());
  EXPECT_EQ(args[2]->cast<Constant>()->as<int64_t>(), 2);
}

TEST(Parser, annotation_inlineTrue)
{
  std::string str = "annotation(inline = true)";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseAnnotation();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 25);

  ASSERT_TRUE((*node)->isa<Annotation>());
  EXPECT_TRUE((*node)->cast<Annotation>()->getInlineProperty());
}

TEST(Parser, annotation_inlineFalse)
{
  std::string str = "annotation(inline = false)";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

  auto node = parser.parseAnnotation();
  ASSERT_TRUE(node.has_value());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 26);

  ASSERT_TRUE((*node)->isa<Annotation>());
  EXPECT_FALSE((*node)->cast<Annotation>()->getInlineProperty());
}

TEST(Parser, annotation_inverseFunction)
{
  std::string str = "annotation(inverse(y = foo1(x, z), z = foo2(x, y)))";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

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

TEST(Parser, annotation_functionDerivativeWithOrder)
{
  std::string str = "annotation(derivative(order=2)=foo1)";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

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

TEST(Parser, annotation_functionDerivativeWithoutOrder)
{
  std::string str = "annotation(derivative=foo1)";

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
  Parser parser(diagnostics, sourceFile);

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
