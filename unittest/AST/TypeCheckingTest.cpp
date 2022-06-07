#include "gtest/gtest.h"
#include "marco/AST/AST.h"
#include "marco/AST/Passes/TypeCheckingPass.h"
#include "marco/Diagnostic/Printer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::diagnostic;

#define LOC SourceRange::unknown()

TEST(TypeChecking, whileStatement_booleanScalarCondition)
{
  auto condition = Expression::constant(LOC, makeType<BuiltInType::Boolean>(), false);
  auto statement = Statement::whileStatement(LOC, std::move(condition), llvm::None);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeCheckingPass pass(diagnostics);
  EXPECT_TRUE(pass.run<Statement>(*statement));
}

TEST(TypeChecking, whileStatement_integerScalarConditionImplicitlyCast)
{
  auto condition = Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0);
  auto statement = Statement::whileStatement(LOC, std::move(condition), llvm::None);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeCheckingPass pass(diagnostics);
  EXPECT_TRUE(pass.run<Statement>(*statement));
}

TEST(TypeChecking, whileStatement_realScalarConditionImplicitlyCast)
{
  auto condition = Expression::constant(LOC, makeType<BuiltInType::Real>(), 0);
  auto statement = Statement::whileStatement(LOC, std::move(condition), llvm::None);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeCheckingPass pass(diagnostics);
  EXPECT_TRUE(pass.run<Statement>(*statement));
}

TEST(TypeChecking, whileStatement_arrayConditionNotAllowed)
{
  std::vector<std::unique_ptr<Expression>> values;

  values.push_back(Expression::constant(LOC, makeType<BuiltInType::Boolean>(), false));
  values.push_back(Expression::constant(LOC, makeType<BuiltInType::Boolean>(), false));

  auto condition = Expression::array(LOC, makeType<BuiltInType::Integer>(2), values);
  auto statement = Statement::whileStatement(LOC, std::move(condition), llvm::None);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeCheckingPass pass(diagnostics);
  EXPECT_FALSE(pass.run<Statement>(*statement));
}
