#include "gtest/gtest.h"
#include "marco/AST/AST.h"
#include "marco/AST/Passes/ConstantFoldingPass.h"
#include "marco/Diagnostic/Printer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::diagnostic;

#define LOC SourceRange::unknown()

//===----------------------------------------------------------------------===//
// Operation: negation
//===----------------------------------------------------------------------===//

TEST(ConstantFolding, neg_integerScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 5));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Integer>(), OperationKind::negate, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Integer>());
  EXPECT_EQ(expression->get<Constant>()->as<BuiltInType::Integer>(), -5);
}

TEST(ConstantFolding, neg_realScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 5.1));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Real>(), OperationKind::negate, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Real>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Real>(), -5.1);
}

//===----------------------------------------------------------------------===//
// Operation: addition
//===----------------------------------------------------------------------===//

TEST(ConstantFolding, add_integerScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 3));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 5));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Integer>(), OperationKind::add, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Integer>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Integer>(), 8);
}

TEST(ConstantFolding, add_realScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 3.1));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 5.2));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Real>(), OperationKind::add, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Real>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Real>(), 8.3);
}

TEST(ConstantFolding, add_integerScalar_realScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 3));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 5.1));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Real>(), OperationKind::add, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Real>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Real>(), 8.1);
}

TEST(ConstantFolding, add_realScalar_integerScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 3.1));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 5));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Real>(), OperationKind::add, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Real>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Real>(), 8.1);
}

//===----------------------------------------------------------------------===//
// Operation: addition
//===----------------------------------------------------------------------===//

TEST(ConstantFolding, sub_integerScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 3));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 5));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Integer>(), OperationKind::subtract, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Integer>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Integer>(), -2);
}

TEST(ConstantFolding, sub_realScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 3.1));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 5.2));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Real>(), OperationKind::subtract, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Real>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Real>(), -2.1);
}

TEST(ConstantFolding, sub_integerScalar_realScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 3));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 5.1));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Real>(), OperationKind::subtract, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Real>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Real>(), -2.1);
}

TEST(ConstantFolding, sub_realScalar_integerScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 3.1));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 5));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Real>(), OperationKind::subtract, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Real>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Real>(), -1.9);
}

//===----------------------------------------------------------------------===//
// Operation: multiplication
//===----------------------------------------------------------------------===//

TEST(ConstantFolding, mul_integerScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 2));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 3));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Integer>(), OperationKind::multiply, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Integer>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Integer>(), 6);
}

TEST(ConstantFolding, mul_realScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 2.1));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 3.2));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Real>(), OperationKind::multiply, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Real>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Real>(), 6.72);
}

TEST(ConstantFolding, mul_integerScalar_realScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 2));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 3.1));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Real>(), OperationKind::multiply, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Real>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Real>(), 6.2);
}

TEST(ConstantFolding, mul_realScalar_integerScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 2.1));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 3));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Real>(), OperationKind::multiply, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Real>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Real>(), 6.3);
}

//===----------------------------------------------------------------------===//
// Operation: division
//===----------------------------------------------------------------------===//

TEST(ConstantFolding, div_integerScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 6));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 3));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Integer>(), OperationKind::divide, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Integer>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Integer>(), 2);
}

TEST(ConstantFolding, div_realScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 12.9));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 4.3));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Real>(), OperationKind::divide, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Real>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Real>(), 3);
}

TEST(ConstantFolding, div_integerScalar_realScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 3));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0.5));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Real>(), OperationKind::divide, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Real>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Real>(), 6);
}

TEST(ConstantFolding, div_realScalar_integerScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 6.0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 3));

  auto expression = Expression::operation(LOC, makeType<BuiltInType::Real>(), OperationKind::divide, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  ConstantFoldingPass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  ASSERT_TRUE(expression->isa<Constant>());
  EXPECT_TRUE(expression->getType().isa<BuiltInType::Real>());
  EXPECT_DOUBLE_EQ(expression->get<Constant>()->as<BuiltInType::Real>(), 2);
}
