#include "gtest/gtest.h"
#include "marco/AST/AST.h"
#include "marco/AST/Passes/TypeInferencePass.h"
#include "marco/Diagnostic/Printer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::diagnostic;

#define LOC SourceRange::unknown()

//===----------------------------------------------------------------------===//
// Operation: negation
//===----------------------------------------------------------------------===//

TEST(TypeInference, neg_integerScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::negate, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>());
}

TEST(TypeInference, neg_realScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::negate, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, neg_1DArray)
{
  std::vector<std::unique_ptr<Expression>> args;

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Integer>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::negate, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>(3));
}

//===----------------------------------------------------------------------===//
// Operation: addition
//===----------------------------------------------------------------------===//

TEST(TypeInference, add_integerScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::add, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>());
}

TEST(TypeInference, add_realScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::add, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, add_integerScalar_realScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::add, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, add_realScalar_integerScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::add, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, add_1DArrays)
{
  std::vector<std::unique_ptr<Expression>> args;

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Integer>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Real>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::add, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>(3));
}

TEST(TypeInference, add_incompatibleRanks)
{
  std::vector<std::unique_ptr<Expression>> args;

  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Real>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::add, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  EXPECT_FALSE(pass.run<Expression>(*expression));
}

TEST(TypeInference, add_incompatibleDimensions)
{
  std::vector<std::unique_ptr<Expression>> args;

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Integer>(2),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Real>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Real>(), 0)
      })));

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Real>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::add, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  EXPECT_FALSE(pass.run<Expression>(*expression));
}

//===----------------------------------------------------------------------===//
// Operation: element-wise addition
//===----------------------------------------------------------------------===//

TEST(TypeInference, addEW_integerScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::addEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>());
}

TEST(TypeInference, addEW_realScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::addEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, addEW_integerScalar_realScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::addEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, addEW_realScalar_integerScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::addEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, addEW_scalar_1DArray)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  
  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Integer>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));
  
  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::addEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>(3));
}

TEST(TypeInference, addEW_1DArray_scalar)
{
  std::vector<std::unique_ptr<Expression>> args;

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Integer>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::addEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>(3));
}

TEST(TypeInference, addEW_1DArrays)
{
  std::vector<std::unique_ptr<Expression>> args;

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Integer>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Real>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::addEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>(3));
}

//===----------------------------------------------------------------------===//
// Operation: subtraction
//===----------------------------------------------------------------------===//

TEST(TypeInference, sub_integerScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::subtract, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>());
}

TEST(TypeInference, sub_realScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::subtract, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, sub_integerScalar_realScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::subtract, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, sub_realScalar_integerScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::subtract, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, sub_1DArrays)
{
  std::vector<std::unique_ptr<Expression>> args;

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Integer>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Real>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::subtract, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>(3));
}

TEST(TypeInference, sub_incompatibleRanks)
{
  std::vector<std::unique_ptr<Expression>> args;

  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Real>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::subtract, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  EXPECT_FALSE(pass.run<Expression>(*expression));
}

TEST(TypeInference, sub_incompatibleDimensions)
{
  std::vector<std::unique_ptr<Expression>> args;

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Integer>(2),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Real>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Real>(), 0)
      })));

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Real>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::subtract, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  EXPECT_FALSE(pass.run<Expression>(*expression));
}

//===----------------------------------------------------------------------===//
// Operation: element-wise subtraction
//===----------------------------------------------------------------------===//

TEST(TypeInference, subEW_integerScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::subtractEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>());
}

TEST(TypeInference, subEW_realScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::subtractEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, subEW_integerScalar_realScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::subtractEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, subEW_realScalar_integerScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::subtractEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, subEW_scalar_1DArray)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Integer>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::subtractEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>(3));
}

TEST(TypeInference, subEW_1DArray_scalar)
{
  std::vector<std::unique_ptr<Expression>> args;

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Integer>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::subtractEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>(3));
}

TEST(TypeInference, subEW_1DArrays)
{
  std::vector<std::unique_ptr<Expression>> args;

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Integer>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Real>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::subtractEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>(3));
}

//===----------------------------------------------------------------------===//
// Operation: multiplication
//===----------------------------------------------------------------------===//

TEST(TypeInference, mul_integerScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::multiply, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>());
}

TEST(TypeInference, mul_realScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::multiply, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, mul_integerScalar_realScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::multiply, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, mul_realScalar_integerScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::multiply, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, mul_1DArrays)
{
  std::vector<std::unique_ptr<Expression>> args;

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Integer>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  args.push_back(Expression::array(
      LOC, makeType<BuiltInType::Real>(3),
      llvm::makeArrayRef({
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0),
          Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0)
      })));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::multiply, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>());
}

// TODO tests

//===----------------------------------------------------------------------===//
// Operation: element-wise multiplication
//===----------------------------------------------------------------------===//

TEST(TypeInference, mulEW_integerScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::multiplyEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>());
}

TEST(TypeInference, mulEW_realScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::multiplyEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, mulEW_integerScalar_realScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::multiplyEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, mulEW_realScalar_integerScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::multiplyEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

// TODO tests

//===----------------------------------------------------------------------===//
// Operation: division
//===----------------------------------------------------------------------===//

TEST(TypeInference, div_integerScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::divide, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>());
}

TEST(TypeInference, div_realScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::divide, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, div_integerScalar_realScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::divide, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, div_realScalar_integerScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::divide, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

// TODO tests

//===----------------------------------------------------------------------===//
// Operation: element-wise division
//===----------------------------------------------------------------------===//

TEST(TypeInference, divEW_integerScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::divideEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Integer>());
}

TEST(TypeInference, divEW_realScalars)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::divideEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, divEW_integerScalar_realScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::divideEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

TEST(TypeInference, divEW_realScalar_integerScalar)
{
  std::vector<std::unique_ptr<Expression>> args;
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  args.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::operation(LOC, Type::unknown(), OperationKind::divideEW, args);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>());
}

// TODO tests

//===----------------------------------------------------------------------===//
// Array
//===----------------------------------------------------------------------===//

TEST(TypeInference, array)
{
  std::vector<std::unique_ptr<Expression>> values;

  values.push_back(Expression::constant(LOC, makeType<BuiltInType::Boolean>(), false));
  values.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  values.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::array(LOC, Type::unknown(), values);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), makeType<BuiltInType::Real>(3));
}

//===----------------------------------------------------------------------===//
// Tuple
//===----------------------------------------------------------------------===//

TEST(TypeInference, tuple)
{
  std::vector<std::unique_ptr<Expression>> values;

  values.push_back(Expression::constant(LOC, makeType<BuiltInType::Boolean>(), false));
  values.push_back(Expression::constant(LOC, makeType<BuiltInType::Integer>(), 0));
  values.push_back(Expression::constant(LOC, makeType<BuiltInType::Real>(), 0));

  auto expression = Expression::tuple(LOC, Type::unknown(), values);

  DiagnosticEngine diagnostics(std::make_unique<Printer>());
  TypeInferencePass pass(diagnostics);
  ASSERT_TRUE(pass.run<Expression>(*expression));

  EXPECT_EQ(expression->getType(), Type(PackedType({ makeType<BuiltInType::Boolean>(), makeType<BuiltInType::Integer>(), makeType<BuiltInType::Real>() })));
}
