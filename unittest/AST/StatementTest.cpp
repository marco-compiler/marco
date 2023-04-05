#include "gtest/gtest.h"
#include "marco/AST/AST.h"

using namespace ::marco;
using namespace ::marco::ast;

#define LOC SourceRange::unknown()

/*
TEST(AST, ifStatement_emptyBody)
{
	auto condition = std::make_unique<Constant>(LOC);
  condition->setValue(true);

  auto node = std::make_unique<IfStatement>(LOC);
  node->setIfCondition(std::move(condition));

	EXPECT_EQ(block.size(), 0);
}

TEST(AST, ifStatement_nonEmptyBody)
{
	auto condition = std::make_unique<Constant>(LOC, true);

	auto statement1 = Statement::assignmentStatement(
			LOC,
			Expression::reference(LOC, makeType<BuiltInType::Integer>(), "x"),
			Expression::constant(LOC,  makeType<BuiltInType::Integer>(), 1));

	auto statement2 = Statement::assignmentStatement(
      LOC,
			Expression::reference(LOC, makeType<BuiltInType::Integer>(), "y"),
			Expression::constant(LOC, makeType<BuiltInType::Integer>(), 2));

	IfStatement::Block block(
			std::move(condition),
			llvm::makeArrayRef({
					std::move(statement1),
					std::move(statement2)
			}));

	EXPECT_EQ(block.size(), 2);
}

TEST(AST, forStatement_emptyBody)
{
	auto condition = Expression::constant(LOC, makeType<BuiltInType::Boolean>(), true);

	auto induction = Induction::build(
      LOC, "i",
			Expression::constant(LOC, makeType<BuiltInType::Integer>(), 1),
			Expression::constant(LOC, makeType<BuiltInType::Integer>(), 2),
      Expression::constant(LOC, makeType<BuiltInType::Integer>(), 1));

	auto statement = Statement::forStatement(LOC, std::move(induction), llvm::None);

	EXPECT_EQ(statement->get<ForStatement>()->size(), 0);
}

TEST(AST, forStatement_nonEmptyBody)
{
	auto condition = Expression::constant(LOC, makeType<BuiltInType::Boolean>(), true);

	auto induction = Induction::build(
      LOC, "i",
			Expression::constant(LOC, makeType<BuiltInType::Integer>(), 1),
			Expression::constant(LOC, makeType<BuiltInType::Integer>(), 2),
      Expression::constant(LOC, makeType<BuiltInType::Integer>(), 1));

	auto statement1 = Statement::assignmentStatement(
      LOC,
			Expression::reference(LOC, makeType<BuiltInType::Integer>(), "x"),
			Expression::constant(LOC, makeType<BuiltInType::Integer>(), 1));

	auto statement2 = Statement::assignmentStatement(
      LOC,
			Expression::reference(LOC, makeType<BuiltInType::Integer>(), "y"),
			Expression::constant(LOC, makeType<BuiltInType::Integer>(), 2));

	auto statement = Statement::forStatement(
      LOC,
			std::move(induction),
			llvm::makeArrayRef({
					std::move(statement1),
					std::move(statement2)
			}));

	EXPECT_EQ(statement->get<ForStatement>()->size(), 2);
}

TEST(AST, whileStatement_emptyBody)
{
	auto condition = Expression::constant(LOC, makeType<BuiltInType::Boolean>(), true);
	auto statement = Statement::whileStatement(LOC, std::move(condition), llvm::None);

	EXPECT_EQ(statement->get<WhileStatement>()->size(), 0);
}

TEST(AST, whileStatement_nonEmptyBody)
{
	auto condition = Expression::constant(LOC, makeType<BuiltInType::Boolean>(), true);

	auto statement1 = Statement::assignmentStatement(
      LOC,
			Expression::reference(LOC, makeType<BuiltInType::Integer>(), "x"),
			Expression::constant(LOC, makeType<BuiltInType::Integer>(), 1));

	auto statement2 = Statement::assignmentStatement(
      LOC,
			Expression::reference(LOC, makeType<BuiltInType::Integer>(), "y"),
			Expression::constant(LOC, makeType<BuiltInType::Integer>(), 2));

	auto statement = Statement::whileStatement(
      LOC,
			std::move(condition),
			llvm::makeArrayRef({
					std::move(statement1),
					std::move(statement2)
			}));

	EXPECT_EQ(statement->get<WhileStatement>()->size(), 2);
}
*/