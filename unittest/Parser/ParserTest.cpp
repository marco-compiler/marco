#include "gtest/gtest.h"
#include "marco/Diagnostic/Printer.h"
#include "marco/Parser/Parser.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::diagnostic;
using namespace ::marco::parser;

TEST(Parser, rawValue_true)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "true";

  Parser parser(diagnostics, str);
  auto node = parser.parseBoolValue();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ(node->getValue(), true);

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 4);
}

TEST(Parser, rawValue_false)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "false";

  Parser parser(diagnostics, str);
  auto node = parser.parseBoolValue();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ(node->getValue(), false);

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 5);
}

TEST(Parser, rawValue_integer)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "012345";

  Parser parser(diagnostics, str);
  auto node = parser.parseIntValue();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ(node->getValue(), 12345);

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 6);
}

TEST(Parser, rawValue_float)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "1.23";

  Parser parser(diagnostics, str);
  auto node = parser.parseFloatValue();
  ASSERT_TRUE(node.hasValue());

  EXPECT_DOUBLE_EQ(node->getValue(), 1.23);

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 4);
}

TEST(Parser, rawValue_string)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "\"test\"";

  Parser parser(diagnostics, str);
  auto node = parser.parseString();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ(node->getValue(), "test");

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 6);
}

TEST(Parser, identifier)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x.y";

  Parser parser(diagnostics, str);
  auto node = parser.parseIdentifier();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ(node->getLocation().begin.line, 1);
  EXPECT_EQ(node->getLocation().begin.column, 1);

  EXPECT_EQ(node->getLocation().end.line, 1);
  EXPECT_EQ(node->getLocation().end.column, 3);

  EXPECT_EQ(node->getValue(), "x.y");
}

TEST(Parser, algorithm_emptyBody)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "algorithm";

  Parser parser(diagnostics, str);
  auto node = parser.parseAlgorithmSection();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 9);

  EXPECT_TRUE((*node)->getBody().empty());
}

TEST(Parser, model)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "model M\n"
                    "  Real x;\n"
                    "  Real y;\n"
                    "equation\n"
                    "  y = x;"
                    "end M;";

  Parser parser(diagnostics, str);
  auto node = parser.parseClassDefinition();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 7);

  EXPECT_TRUE((*node)->isa<Model>());
  EXPECT_EQ((*node)->get<Model>()->getName(), "M");
}

TEST(Parser, standardFunction)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "function foo\n"
                    "  input Real x;\n"
                    "  output Real y;\n"
                    "algorithm\n"
                    "  y := x;\n"
                    "end foo;";

  Parser parser(diagnostics, str);
  auto node = parser.parseClassDefinition();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 12);

  EXPECT_TRUE((*node)->isa<StandardFunction>());
  EXPECT_EQ((*node)->get<StandardFunction>()->getName(), "foo");
}

TEST(Parser, algorithm_statementsCount)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "algorithm\n"
                    "	x := 1;\n"
                    "	y := 2;\n"
                    "	z := 3;";

  Parser parser(diagnostics, str);
  auto node = parser.parseAlgorithmSection();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 4);
  EXPECT_EQ((*node)->getLocation().end.column, 8);

  EXPECT_EQ((*node)->getBody().size(), 3);
}

TEST(Parser, equation)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "y = x";

  Parser parser(diagnostics, str);
  auto node = parser.parseEquation();
  ASSERT_TRUE(node.hasValue());

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
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "y := x";

  Parser parser(diagnostics, str);
  auto node = parser.parseStatement();
  ASSERT_TRUE(node.hasValue());

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
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "(x, y) := Foo()";

  Parser parser(diagnostics, str);
  auto node = parser.parseStatement();
  ASSERT_TRUE(node.hasValue());

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
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "(x, , z) := Foo()";

  Parser parser(diagnostics, str);
  auto node = parser.parseStatement();
  ASSERT_TRUE(node.hasValue());

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

TEST(Parser, statement_break)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "break";

  Parser parser(diagnostics, str);
  auto node = parser.parseStatement();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  EXPECT_TRUE((*node)->isa<BreakStatement>());
}

TEST(Parser, statement_return)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "return";

  Parser parser(diagnostics, str);
  auto node = parser.parseStatement();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  EXPECT_TRUE((*node)->isa<ReturnStatement>());
}

TEST(Parser, expression_constant)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "012345";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Constant>());
  EXPECT_EQ((*node)->get<Constant>()->get<BuiltInType::Integer>(), 12345);
}

TEST(Parser, expression_not)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "not x";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::negate);
}

TEST(Parser, expression_and)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x and y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 7);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::land);
}

TEST(Parser, expression_or)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x or y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::lor);
}

TEST(Parser, expression_equal)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x == y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::equal);
}

TEST(Parser, expression_notEqual)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x <> y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::different);
}

TEST(Parser, expression_less)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x < y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::less);
}

TEST(Parser, expression_lessEqual)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x <= y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::lessEqual);
}

TEST(Parser, expression_greater)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x > y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::greater);
}

TEST(Parser, expression_greaterEqual)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x >= y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::greaterEqual);
}

TEST(Parser, expression_addition)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x + y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::add);
}

TEST(Parser, expression_additionElementWise)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x .+ y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::addEW);
}

TEST(Parser, expression_subtraction)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x - y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::subtract);
}

TEST(Parser, expression_subtractionElementWise)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x .- y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::subtractEW);
}

TEST(Parser, expression_multiplication)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x * y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::multiply);
}

TEST(Parser, expression_multiplicationElementWise)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x .* y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::multiplyEW);
}

TEST(Parser, expression_division)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x / y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::divide);
}

TEST(Parser, expression_divisionElementWise)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x ./ y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::divideEW);
}

TEST(Parser, expression_pow)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x ^ y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 5);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::powerOf);
}

TEST(Parser, expression_powElementWise)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "x .^ y";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 6);

  ASSERT_TRUE((*node)->isa<Operation>());
  EXPECT_EQ((*node)->get<Operation>()->getOperationKind(), OperationKind::powerOfEW);
}

TEST(Parser, expression_tuple_empty)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "()";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 2);

  ASSERT_TRUE((*node)->isa<Tuple>());
  EXPECT_EQ((*node)->get<Tuple>()->size(), 0);
}

TEST(Parser, expression_componentReference)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "var";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 3);

  ASSERT_TRUE((*node)->isa<ReferenceAccess>());
  EXPECT_EQ((*node)->get<ReferenceAccess>()->getName(), "var");
}

TEST(Parser, expression_array)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "{1, 2, 3, 4, 5}";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

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
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "var[3,i,:]";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

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
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "{1, 2, 3, 4, 5}[i]";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

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
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "foo()[i]";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

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
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "foo()";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

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
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "foo(x, 1, 2)";

  Parser parser(diagnostics, str);
  auto node = parser.parseExpression();
  ASSERT_TRUE(node.hasValue());

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
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "annotation(inline = true)";

  Parser parser(diagnostics, str);
  auto node = parser.parseAnnotation();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 25);

  EXPECT_TRUE((*node)->getInlineProperty());
}

TEST(Parser, annotation_inlineFalse)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "annotation(inline = false)";

  Parser parser(diagnostics, str);
  auto node = parser.parseAnnotation();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 26);

  EXPECT_FALSE((*node)->getInlineProperty());
}

TEST(Parser, annotation_inverseFunction)
{
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "annotation(inverse(y = foo1(x, z), z = foo2(x, y)))";

  Parser parser(diagnostics, str);
  auto node = parser.parseAnnotation();
  ASSERT_TRUE(node.hasValue());

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
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "annotation(derivative(order=2)=foo1)";

  Parser parser(diagnostics, str);
  auto node = parser.parseAnnotation();
  ASSERT_TRUE(node.hasValue());

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
  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  std::string str = "annotation(derivative=foo1)";

  Parser parser(diagnostics, str);
  auto node = parser.parseAnnotation();
  ASSERT_TRUE(node.hasValue());

  EXPECT_EQ((*node)->getLocation().begin.line, 1);
  EXPECT_EQ((*node)->getLocation().begin.column, 1);

  EXPECT_EQ((*node)->getLocation().end.line, 1);
  EXPECT_EQ((*node)->getLocation().end.column, 27);

  auto annotation = (*node)->getDerivativeAnnotation();

  EXPECT_EQ(annotation.getName(), "foo1");
  EXPECT_EQ(annotation.getOrder(), 1);
}
