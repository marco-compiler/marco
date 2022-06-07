#include "gtest/gtest.h"
#include "marco/AST/AST.h"
#include "marco/AST/Passes/SemanticAnalysisPass.h"
#include "marco/Diagnostic/Printer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::diagnostic;

#define LOC SourceRange::unknown()

// function foo
//   Real x;
// algorithm
//   x := 0;
// end foo;
TEST(SemanticAnalysis, function_publicMembersMustBeInputOrOutput)
{
  std::vector<std::unique_ptr<Member>> members;

  members.push_back(Member::build(
      LOC, "x",
      makeType<BuiltInType::Integer>(),
      TypePrefix::none()));

  std::vector<std::unique_ptr<Statement>> statements;

  statements.push_back(Statement::assignmentStatement(
      LOC,
      Expression::reference(LOC, makeType<BuiltInType::Integer>(), "x"),
      Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)));

  auto cls = Class::standardFunction(
      LOC, true, "foo",
      members,
      Algorithm::build(LOC, llvm::None));

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  SemanticAnalysisPass pass(diagnostics);

  EXPECT_FALSE(pass.run<Class>(*cls));
}

// function foo
//   input Real x;
// algorithm
//   x := 0;
// end foo;
TEST(SemanticAnalysis, function_assignmentToInputMember)
{
  std::vector<std::unique_ptr<Member>> members;

  members.push_back(Member::build(
      LOC, "x",
      makeType<BuiltInType::Integer>(),
      TypePrefix(ParameterQualifier::none, IOQualifier::input)));

  std::vector<std::unique_ptr<Statement>> statements;

  statements.push_back(Statement::assignmentStatement(
      LOC,
      Expression::reference(LOC, makeType<BuiltInType::Integer>(), "x"),
      Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)));

  auto cls = Class::standardFunction(
      LOC, true, "foo",
      members,
      Algorithm::build(LOC, statements));

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  SemanticAnalysisPass pass(diagnostics);

  EXPECT_FALSE(pass.run<Class>(*cls));
}

TEST(SemanticAnalysis, assignmentStatement_destinationsMustBeLValues)
{
  auto statement = Statement::assignmentStatement(
      LOC,
      Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
      Expression::constant(LOC, makeType<BuiltInType::Integer>(), 1));

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  SemanticAnalysisPass pass(diagnostics);

  EXPECT_FALSE(pass.run<Statement>(*statement));
}
