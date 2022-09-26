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

  EXPECT_TRUE((*node)->getBody().empty());
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

  EXPECT_TRUE((*node)->isa<Model>());
  EXPECT_EQ((*node)->get<Model>()->getName(), "M");
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

  EXPECT_TRUE((*node)->isa<StandardFunction>());
  auto function = (*node)->get<StandardFunction>();

  EXPECT_EQ(function->getName(), "foo");
  ASSERT_EQ(function->getMembers().size(), 2);
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

  EXPECT_TRUE((*node)->isa<PartialDerFunction>());
  auto function = (*node)->get<PartialDerFunction>();

  EXPECT_EQ(function->getDerivedFunction()->get<ReferenceAccess>()->getName(), "Foo");
  EXPECT_EQ(function->getIndependentVariables().size(), 2);
  EXPECT_EQ(function->getIndependentVariables()[0]->get<ReferenceAccess>()->getName(), "x");
  EXPECT_EQ(function->getIndependentVariables()[1]->get<ReferenceAccess>()->getName(), "y");
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

  EXPECT_EQ((*node)->getBody().size(), 3);
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

  auto lhs = (*node)->getLhsExpression();
  ASSERT_TRUE(lhs->isa<ReferenceAccess>());
  EXPECT_EQ(lhs->get<ReferenceAccess>()->getName(), "y");

  auto rhs = (*node)->getRhsExpression();
  ASSERT_TRUE(rhs->isa<ReferenceAccess>());
  EXPECT_EQ(rhs->get<ReferenceAccess>()->getName(), "x");
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

  auto statement = (*node)->get<AssignmentStatement>();

  ASSERT_TRUE(statement->getDestinations()->isa<Tuple>());
  auto destinations = statement->getDestinations()->get<Tuple>();
  ASSERT_EQ(destinations->size(), 1);
  ASSERT_TRUE((*destinations)[0]->isa<ReferenceAccess>());
  EXPECT_EQ((*destinations)[0]->get<ReferenceAccess>()->getName(), "y");

  ASSERT_TRUE(statement->getExpression()->isa<ReferenceAccess>());
  EXPECT_EQ(statement->getExpression()->get<ReferenceAccess>()->getName(), "x");
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
  auto statement = (*node)->get<AssignmentStatement>();

  ASSERT_TRUE(statement->getDestinations()->isa<Tuple>());
  auto* destinations = statement->getDestinations()->get<Tuple>();

  ASSERT_EQ(destinations->size(), 2);
  EXPECT_EQ(destinations->getArg(0)->get<ReferenceAccess>()->getName(), "x");
  EXPECT_EQ(destinations->getArg(1)->get<ReferenceAccess>()->getName(), "y");
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
  auto statement = (*node)->get<AssignmentStatement>();

  ASSERT_TRUE(statement->getDestinations()->isa<Tuple>());
  auto* destinations = statement->getDestinations()->get<Tuple>();

  ASSERT_EQ(destinations->size(), 3);
  EXPECT_EQ(destinations->getArg(0)->get<ReferenceAccess>()->getName(), "x");
  EXPECT_TRUE(destinations->getArg(1)->get<ReferenceAccess>()->isDummy());
  EXPECT_EQ(destinations->getArg(2)->get<ReferenceAccess>()->getName(), "z");
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
  auto statement = (*node)->get<IfStatement>();

  ASSERT_EQ(statement->size(), 1);
  EXPECT_EQ(statement->getBlock(0).size(), 2);
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
  auto statement = (*node)->get<IfStatement>();

  ASSERT_EQ(statement->size(), 2);
  EXPECT_EQ(statement->getBlock(0).size(), 2);
  EXPECT_EQ(statement->getBlock(1).size(), 1);
}

TEST(Parser, statement_ifElseIfElse)
{
  std::string str = "if false then\n"
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

  EXPECT_EQ((*node)->getLocation().end.line, 10);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<IfStatement>());
  auto statement = (*node)->get<IfStatement>();

  ASSERT_EQ(statement->size(), 3);
  EXPECT_EQ(statement->getBlock(0).size(), 3);
  EXPECT_EQ(statement->getBlock(1).size(), 2);
  EXPECT_EQ(statement->getBlock(2).size(), 1);
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
  auto statement = (*node)->get<ForStatement>();

  ASSERT_EQ(statement->size(), 2);
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
  auto statement = (*node)->get<WhileStatement>();

  ASSERT_EQ(statement->size(), 2);
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
  EXPECT_EQ((*node)->get<Constant>()->get<BuiltInType::Integer>(), 12345);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::lnot);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::land);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::lor);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::equal);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::different);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::less);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::lessEqual);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::greater);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::greaterEqual);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::add);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::addEW);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::subtract);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::subtractEW);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::multiply);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::multiplyEW);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::divide);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::divideEW);
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
  ASSERT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::add);

  auto* addition = (*node)->get<Operation>();

  auto* x = addition->getArg(0);
  ASSERT_TRUE(x->isa<ReferenceAccess>());
  EXPECT_EQ(x->get<ReferenceAccess>()->getName(), "x");

  auto* multiplication = addition->getArg(1)->get<Operation>();
  ASSERT_EQ(multiplication->getOperationKind(), OperationKind::multiply);

  auto* y = multiplication->getArg(0);
  ASSERT_TRUE(y->isa<ReferenceAccess>());
  EXPECT_EQ(y->get<ReferenceAccess>()->getName(), "y");

  auto* z = multiplication->getArg(1);
  ASSERT_TRUE(z->isa<ReferenceAccess>());
  EXPECT_EQ(z->get<ReferenceAccess>()->getName(), "z");
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
  ASSERT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::add);

  auto* addition = (*node)->get<Operation>();

  auto* z = addition->getArg(1);
  ASSERT_TRUE(z->isa<ReferenceAccess>());
  EXPECT_EQ(z->get<ReferenceAccess>()->getName(), "z");

  ASSERT_TRUE(addition->getArg(0)->isa<Operation>());

  auto* multiplication = addition->getArg(0)->get<Operation>();
  ASSERT_EQ(multiplication->getOperationKind(), OperationKind::multiply);

  auto* x = multiplication->getArg(0);
  ASSERT_TRUE(x->isa<ReferenceAccess>());
  EXPECT_EQ(x->get<ReferenceAccess>()->getName(), "x");

  auto* y = multiplication->getArg(1);
  ASSERT_TRUE(y->isa<ReferenceAccess>());
  EXPECT_EQ(y->get<ReferenceAccess>()->getName(), "y");
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

  auto* division = (*node)->get<Operation>();
  ASSERT_EQ(division->getOperationKind(), OperationKind::divide);

  auto* z = division->getArg(1);
  ASSERT_TRUE(z->isa<ReferenceAccess>());
  EXPECT_EQ(z->get<ReferenceAccess>()->getName(), "z");

  ASSERT_TRUE(division->getArg(0)->isa<Operation>());

  auto* multiplication = division->getArg(0)->get<Operation>();
  ASSERT_EQ(multiplication->getOperationKind(), OperationKind::multiply);

  auto* x = multiplication->getArg(0);
  ASSERT_TRUE(x->isa<ReferenceAccess>());
  EXPECT_EQ(x->get<ReferenceAccess>()->getName(), "x");

  auto* y = multiplication->getArg(1);
  ASSERT_TRUE(y->isa<ReferenceAccess>());
  EXPECT_EQ(y->get<ReferenceAccess>()->getName(), "y");
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

  auto* multiplication = (*node)->get<Operation>();
  ASSERT_EQ(multiplication->getOperationKind(), OperationKind::multiply);

  auto* z = multiplication->getArg(1);
  ASSERT_TRUE(z->isa<ReferenceAccess>());
  EXPECT_EQ(z->get<ReferenceAccess>()->getName(), "z");

  ASSERT_TRUE(multiplication->getArg(0)->isa<Operation>());

  auto* division = multiplication->getArg(0)->get<Operation>();
  ASSERT_EQ(division->getOperationKind(), OperationKind::divide);

  auto* x = division->getArg(0);
  ASSERT_TRUE(x->isa<ReferenceAccess>());
  EXPECT_EQ(x->get<ReferenceAccess>()->getName(), "x");

  auto* y = division->getArg(1);
  ASSERT_TRUE(y->isa<ReferenceAccess>());
  EXPECT_EQ(y->get<ReferenceAccess>()->getName(), "y");
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

  auto* division = (*node)->get<Operation>();
  ASSERT_EQ(division->getOperationKind(), OperationKind::divide);

  auto* x = division->getArg(0);
  ASSERT_TRUE(x->isa<ReferenceAccess>());
  EXPECT_EQ(x->get<ReferenceAccess>()->getName(), "x");

  ASSERT_TRUE(division->getArg(1)->isa<Operation>());

  auto* multiplication = division->getArg(1)->get<Operation>();
  ASSERT_EQ(multiplication->getOperationKind(), OperationKind::multiply);

  auto* y = multiplication->getArg(0);
  ASSERT_TRUE(y->isa<ReferenceAccess>());
  EXPECT_EQ(y->get<ReferenceAccess>()->getName(), "y");

  auto* z = multiplication->getArg(1);
  ASSERT_TRUE(z->isa<ReferenceAccess>());
  EXPECT_EQ(z->get<ReferenceAccess>()->getName(), "z");
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::powerOf);
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
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::powerOfEW);
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
  EXPECT_EQ((*node)->get<Tuple>()->size(), 0);
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

  ASSERT_TRUE((*node)->isa<ReferenceAccess>());
  EXPECT_EQ((*node)->get<ReferenceAccess>()->getName(), "var");
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
  const auto& array = *(*node)->get<Array>();
  ASSERT_EQ(array.size(), 5);

  ASSERT_TRUE(array[0]->isa<Constant>());
  EXPECT_EQ(array[0]->get<Constant>()->get<BuiltInType::Integer>(), 1);

  ASSERT_TRUE(array[1]->isa<Constant>());
  EXPECT_EQ(array[1]->get<Constant>()->get<BuiltInType::Integer>(), 2);

  ASSERT_TRUE(array[2]->isa<Constant>());
  EXPECT_EQ(array[2]->get<Constant>()->get<BuiltInType::Integer>(), 3);

  ASSERT_TRUE(array[3]->isa<Constant>());
  EXPECT_EQ(array[3]->get<Constant>()->get<BuiltInType::Integer>(), 4);

  ASSERT_TRUE(array[4]->isa<Constant>());
  EXPECT_EQ(array[4]->get<Constant>()->get<BuiltInType::Integer>(), 5);
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
  auto args =(*node)->get<Operation>()->getArguments();

  ASSERT_TRUE(args[0]->isa<ReferenceAccess>());
  EXPECT_EQ(args[0]->get<ReferenceAccess>()->getName(), "var");

  ASSERT_TRUE(args[1]->isa<Constant>());
  EXPECT_EQ(args[1]->get<Constant>()->get<BuiltInType::Integer>(), 3);

  ASSERT_TRUE(args[2]->isa<ReferenceAccess>());
  EXPECT_EQ(args[2]->get<ReferenceAccess>()->getName(), "i");

  ASSERT_TRUE(args[3]->isa<Constant>());
  EXPECT_EQ(args[3]->get<Constant>()->get<BuiltInType::Integer>(), -1);
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
  ASSERT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::subscription);

  auto args = (*node)->get<Operation>()->getArguments();

  EXPECT_TRUE(args[0]->isa<Array>());
  ASSERT_TRUE(args[1]->isa<ReferenceAccess>());
  EXPECT_EQ(args[1]->get<ReferenceAccess>()->getName(), "i");
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
  ASSERT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::subscription);

  auto args = (*node)->get<Operation>()->getArguments();

  EXPECT_TRUE(args[0]->isa<Call>());
  ASSERT_TRUE(args[1]->isa<ReferenceAccess>());
  EXPECT_EQ(args[1]->get<ReferenceAccess>()->getName(), "i");
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
  auto call = (*node)->get<Call>();

  ASSERT_TRUE(call->getFunction()->isa<ReferenceAccess>());
  EXPECT_EQ(call->getFunction()->get<ReferenceAccess>()->getName(), "foo");

  auto args = call->getArgs();
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
  auto call = (*node)->get<Call>();

  ASSERT_TRUE(call->getFunction()->isa<ReferenceAccess>());
  EXPECT_EQ(call->getFunction()->get<ReferenceAccess>()->getName(), "foo");

  auto args = call->getArgs();
  EXPECT_EQ(args.size(), 3);

  ASSERT_TRUE(args[0]->isa<ReferenceAccess>());
  EXPECT_EQ(args[0]->get<ReferenceAccess>()->getName(), "x");

  ASSERT_TRUE(args[1]->isa<Constant>());
  EXPECT_EQ(args[1]->get<Constant>()->get<BuiltInType::Integer>(), 1);

  ASSERT_TRUE(args[2]->isa<Constant>());
  EXPECT_EQ(args[2]->get<Constant>()->get<BuiltInType::Integer>(), 2);
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

  EXPECT_TRUE((*node)->getInlineProperty());
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

  EXPECT_FALSE((*node)->getInlineProperty());
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

  auto annotation = (*node)->getInverseFunctionAnnotation();

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

  auto annotation = (*node)->getDerivativeAnnotation();

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

  auto annotation = (*node)->getDerivativeAnnotation();

  EXPECT_EQ(annotation.getName(), "foo1");
  EXPECT_EQ(annotation.getOrder(), 1);
}
