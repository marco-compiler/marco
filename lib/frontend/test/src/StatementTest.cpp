#include <gtest/gtest.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/Passes.h>
#include <modelica/frontend/Parser.hpp>

using namespace modelica;
using namespace frontend;

TEST(AST, ifStatementWithEmptyBody)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();
	Expression condition = Expression::constant(location, makeType<bool>(), true);
	IfStatement::Block block(condition, {});
	EXPECT_EQ(block.size(), 0);
}

TEST(AST, ifStatementWithNonEmptyBody)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();
	Expression condition = Expression::constant(location, makeType<bool>(), true);

	AssignmentStatement statement1(location,
																 Expression::reference(location, makeType<int>(), "x"),
																 Expression::constant(location, makeType<int>(), 1));

	AssignmentStatement statement2(location,
																 Expression::reference(location, makeType<int>(), "y"),
																 Expression::constant(location, makeType<int>(), 2));

	IfStatement::Block block(condition, { statement1, statement2 });
	EXPECT_EQ(block.size(), 2);
}

TEST(AST, forStatementWithEmptyBody)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();
	Expression condition = Expression::constant(location, makeType<bool>(), true);
	Induction induction("i", Expression::constant(location, makeType<int>(), 1), Expression::constant(location, makeType<int>(), 2));
	ForStatement statement(location, induction, {});
	EXPECT_EQ(statement.size(), 0);
}

TEST(AST, forStatementWithNonEmptyBody)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();
	Expression condition = Expression::constant(location, makeType<bool>(), true);
	Induction induction("i", Expression::constant(location, makeType<int>(), 1), Expression::constant(location, makeType<int>(), 2));

	AssignmentStatement statement1(location,
																 Expression::reference(location, makeType<int>(), "x"),
																 Expression::constant(location, makeType<int>(), 1));

	AssignmentStatement statement2(location,
																 Expression::reference(location, makeType<int>(), "y"),
																 Expression::constant(location, makeType<int>(), 2));

	ForStatement statement(location, induction, { statement1, statement2 });
	EXPECT_EQ(statement.size(), 2);
}

TEST(AST, whileStatementWithEmptyBody)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();
	Expression condition = Expression::constant(location, makeType<bool>(), true);
	WhileStatement statement(location, condition, {});
	EXPECT_EQ(statement.size(), 0);
}

TEST(AST, whileStatementWithNonEmptyBody)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();
	Expression condition = Expression::constant(location, makeType<bool>(), true);

	AssignmentStatement statement1(location,
																 Expression::reference(location, makeType<int>(), "x"),
																 Expression::constant(location, makeType<int>(), 1));

	AssignmentStatement statement2(location,
																 Expression::reference(location, makeType<int>(), "y"),
																 Expression::constant(location, makeType<int>(), 2));

	WhileStatement statement(location, condition, { statement1, statement2 });
	EXPECT_EQ(statement.size(), 2);
}

TEST(Parser, assignmentStatementIntegerConstant)	// NOLINT
{
	Parser parser("x := 1;");

	auto ast = parser.assignmentStatement();
	ASSERT_FALSE(!ast);

	// Left-hand side
	auto destinations = ast->getDestinations();
	EXPECT_EQ(destinations.size(), 1);
	EXPECT_EQ(destinations[0].get<ReferenceAccess>().getName(), "x");

	// Right-hand side
	auto& expression = ast->getExpression();
	EXPECT_EQ(expression.get<Constant>().get<BuiltInType::Integer>(), 1);
}

TEST(Parser, assignmentStatementFloatConstant)	// NOLINT
{
	Parser parser("x := 3.14;");

	auto ast = parser.assignmentStatement();
	ASSERT_FALSE(!ast);

	// Left-hand side
	auto destinations = ast->getDestinations();
	EXPECT_EQ(destinations.size(), 1);
	EXPECT_EQ(destinations[0].get<ReferenceAccess>().getName(), "x");

	// Right-hand side
	auto& expression = ast->getExpression();
	EXPECT_FLOAT_EQ(expression.get<Constant>().get<BuiltInType::Float>(), 3.14);
}

TEST(Parser, assignmentStatementReference)	// NOLINT
{
	Parser parser("x := y;");

	auto ast = parser.assignmentStatement();
	ASSERT_FALSE(!ast);

	// Left-hand side
	auto destinations = ast->getDestinations();
	EXPECT_EQ(destinations.size(), 1);
	EXPECT_EQ(destinations[0].get<ReferenceAccess>().getName(), "x");

	// Right-hand side
	auto& expression = ast->getExpression();
	EXPECT_EQ(expression.get<ReferenceAccess>().getName(), "y");
}

TEST(Parser, assignmentStatementFunctionCall)	 // NOLINT
{
	Parser parser("x := Foo (y, z);");

	auto ast = parser.assignmentStatement();
	ASSERT_FALSE(!ast);

	// Left-hand side
	auto destinations = ast->getDestinations();
	EXPECT_EQ(destinations.size(), 1);
	EXPECT_EQ(destinations[0].get<ReferenceAccess>().getName(), "x");

	// Right-hand side
	auto& call = ast->getExpression().get<Call>();

	// Function name
	EXPECT_EQ(call.getFunction().get<ReferenceAccess>().getName(), "Foo");

	// Function parameters
	EXPECT_EQ(call.argumentsCount(), 2);
	EXPECT_EQ(call[0].get<ReferenceAccess>().getName(), "y");
	EXPECT_EQ(call[1].get<ReferenceAccess>().getName(), "z");
}

TEST(Parser, assignmentStatementMultipleOutputs)	// NOLINT
{
	Parser parser("(x, y) := Foo ();");

	auto ast = parser.assignmentStatement();
	ASSERT_FALSE(!ast);

	// Right-hand side is not tested because not so important for this test
	auto destinations = ast->getDestinations();
	EXPECT_EQ(destinations.size(), 2);
	EXPECT_EQ(destinations[0].get<ReferenceAccess>().getName(), "x");
	EXPECT_EQ(destinations[1].get<ReferenceAccess>().getName(), "y");
}

TEST(Parser, assignmentStatementIgnoredOutputs)	 // NOLINT
{
	Parser parser("(x, , z) := Foo ();");

	auto ast = parser.assignmentStatement();
	ASSERT_FALSE(!ast);

	// Right-hand side is not tested because not so important for this test
	auto destinations = ast->getDestinations();
	EXPECT_EQ(3, destinations.size());

	EXPECT_EQ(destinations[0].get<ReferenceAccess>().getName(), "x");
	EXPECT_TRUE(destinations[1].get<ReferenceAccess>().isDummy());
	EXPECT_EQ(destinations[2].get<ReferenceAccess>().getName(), "z");
}

TEST(Parser, ifStatementBlockCondition)	 // NOLINT
{
	Parser parser("if false then"
								"  x := 1;"
								"end if;");

	auto ast = parser.ifStatement();
	ASSERT_FALSE(!ast);

	EXPECT_EQ(ast->size(), 1);
	auto& block = (*ast)[0];
	EXPECT_EQ(block.getCondition().get<Constant>().get<BuiltInType::Boolean>(), false);
}

TEST(Parser, ifStatementThenBranchOnly)	 // NOLINT
{
	Parser parser("if false then"
								"  x := 1;"
								"  y := 2;"
								"end if;");

	auto ast = parser.ifStatement();
	ASSERT_FALSE(!ast);

	EXPECT_EQ(ast->size(), 1);
	auto& block = (*ast)[0];
	EXPECT_EQ(block.size(), 2);
}

TEST(Parser, ifStatementElseBranch)	 // NOLINT
{
	Parser parser("if false then"
								"  x := 1;"
								"  y := 2;"
								"else"
								"  x := 1;"
								"end if;");

	auto ast = parser.ifStatement();
	ASSERT_FALSE(!ast);

	EXPECT_EQ(ast->size(), 2);

	auto& thenBlock = (*ast)[0];
	EXPECT_EQ(thenBlock.size(), 2);

	auto& elseBlock = (*ast)[1];
	EXPECT_EQ(elseBlock.getCondition().get<Constant>().get<BuiltInType::Boolean>(), true);
	EXPECT_EQ(elseBlock.size(), 1);
}

TEST(Parser, ifStatementElseIfBranch)	 // NOLINT
{
	Parser parser("if false then"
								"  x := 1;"
								"  y := 2;"
								"  z := 3;"
								"elseif false then"
								"  x := 1;"
								"  y := 2;"
								"else"
								"  x := 1;"
								"end if;");

	auto ast = parser.ifStatement();
	ASSERT_FALSE(!ast);

	EXPECT_EQ(ast->size(), 3);

	auto& thenBlock = (*ast)[0];
	EXPECT_EQ(thenBlock.size(), 3);

	auto& elseIfBlock = (*ast)[1];
	EXPECT_EQ(elseIfBlock.getCondition().get<Constant>().get<BuiltInType::Boolean>(), false);
	EXPECT_EQ(elseIfBlock.size(), 2);

	auto& elseBlock = (*ast)[2];
	EXPECT_EQ(elseBlock.getCondition().get<Constant>().get<BuiltInType::Boolean>(), true);
	EXPECT_EQ(elseBlock.size(), 1);
}

TEST(Parser, forStatementInduction)	 // NOLINT
{
	Parser parser("for i in 1:10 loop"
								"  x := 1;"
								"end for;");

	auto ast = parser.forStatement();
	ASSERT_FALSE(!ast);

	auto& induction = ast->getInduction();
	EXPECT_EQ(induction.getName(), "i");
	EXPECT_EQ(induction.getBegin().get<Constant>().get<BuiltInType::Integer>(), 1);
	EXPECT_EQ(induction.getEnd().get<Constant>().get<BuiltInType::Integer>(), 10);
}

TEST(Parser, forStatementBody)	 // NOLINT
{
	Parser parser("for i in 1:10 loop"
								"  x := 1;"
								"  y := 2;"
								"end for;");

	auto ast = parser.forStatement();
	ASSERT_FALSE(!ast);

	EXPECT_EQ(ast->size(), 2);
}

TEST(Parser, whileStatementCondition)	 // NOLINT
{
	Parser parser("while false loop"
								"  x := 1;"
								"end while;");

	auto ast = parser.whileStatement();
	ASSERT_FALSE(!ast);

	EXPECT_EQ(ast->getCondition().get<Constant>().get<BuiltInType::Boolean>(), false);
}

TEST(Parser, whileStatementBody)	 // NOLINT
{
	Parser parser("while false loop"
								"  x := 1;"
								"  y := 2;"
								"end while;");

	auto ast = parser.whileStatement();
	ASSERT_FALSE(!ast);

	EXPECT_EQ(ast->size(), 2);
}

TEST(TypeChecker, assignmentStatementConstant)	 // NOLINT
{
	Parser parser("function Foo"
								"  output Integer y;"
								"algorithm"
								"  y := 1;"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	TypeChecker typeChecker;
	EXPECT_TRUE(!typeChecker.run(*ast));

	auto& statement = (*ast->get<Function>().getAlgorithms()[0])[0].get<AssignmentStatement>();
	EXPECT_EQ(statement.getDestinations().size(), 1);
	EXPECT_EQ(statement.getDestinations()[0].getType(), makeType<int>());
	EXPECT_EQ(statement.getExpression().getType(), makeType<int>());
}

TEST(TypeChecker, assignmentStatementReference)	 // NOLINT
{
	Parser parser("function Foo"
								"  input Integer x;"
								"  output Integer y;"
								"algorithm"
								"  y := x;"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	TypeChecker typeChecker;
	EXPECT_TRUE(!typeChecker.run(*ast));

	auto& statement = (*ast->get<Function>().getAlgorithms()[0])[0].get<AssignmentStatement>();
	EXPECT_EQ(statement.getDestinations().size(), 1);
	EXPECT_EQ(statement.getDestinations()[0].getType(), makeType<int>());
	EXPECT_EQ(statement.getExpression().getType(), makeType<int>());
}

TEST(TypeChecker, assignmentStatementOperation)	 // NOLINT
{
	Parser parser("function Foo"
								"  input Integer x;"
								"  input Real y;"
								"  output Integer z;"
								"algorithm"
								"  z := x + y;"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	TypeChecker typeChecker;
	EXPECT_TRUE(!typeChecker.run(*ast));

	auto& statement = (*ast->get<Function>().getAlgorithms()[0])[0].get<AssignmentStatement>();
	EXPECT_EQ(statement.getDestinations().size(), 1);
	EXPECT_EQ(statement.getDestinations()[0].getType(), makeType<int>());
	EXPECT_EQ(statement.getExpression().getType(), makeType<float>());
}

TEST(TypeChecker, assignmentStatementCall)	 // NOLINT
{
	Parser parser("function Foo"
								"  input Integer x;"
								"  input Real y;"
								"  output Integer z;"
								"algorithm"
								"  z := Foo(z, z);"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	TypeChecker typeChecker;
	EXPECT_TRUE(!typeChecker.run(*ast));

	auto& statement = (*ast->get<Function>().getAlgorithms()[0])[0].get<AssignmentStatement>();
	EXPECT_EQ(statement.getDestinations().size(), 1);
	EXPECT_EQ(statement.getDestinations()[0].getType(), makeType<int>());
	EXPECT_EQ(statement.getExpression().getType(), makeType<int>());
}
