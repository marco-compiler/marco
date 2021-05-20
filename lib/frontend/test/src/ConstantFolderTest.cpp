#include <gtest/gtest.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/Parser.h>
#include <modelica/frontend/Passes.h>
#include <modelica/frontend/SymbolTable.hpp>

using namespace modelica;
using namespace frontend;
using namespace std;

TEST(folderTest, sumShouldFold)
{
	auto expression = Expression::operation(
			SourceRange::unknown(),
			makeType<int>(),
			OperationKind::add,
			llvm::ArrayRef({
					Expression::constant(SourceRange::unknown(), makeType<int>(), 3),
					Expression::constant(SourceRange::unknown(), makeType<int>(), 4)
			}));

	ConstantFolder folder;

	if (folder.run<Expression>(*expression))
		FAIL();

	EXPECT_TRUE(expression->isa<Constant>());
	EXPECT_EQ(expression->get<Constant>()->get<BuiltInType::Integer>(), 7);
}

TEST(folderTest, subShouldFold)
{
	auto expression = Expression::operation(
			SourceRange::unknown(),
			makeType<int>(),
			OperationKind::subtract,
			llvm::ArrayRef({
					Expression::constant(SourceRange::unknown(), makeType<int>(), 3),
					Expression::constant(SourceRange::unknown(), makeType<int>(), 2)
			}));

	ConstantFolder folder;

	if (folder.run<Expression>(*expression))
		FAIL();

	EXPECT_TRUE(expression->isa<Constant>());
	EXPECT_EQ(expression->get<Constant>()->get<BuiltInType::Integer>(), 1);
}

TEST(folderTest, sumOfSubShouldFold)
{
	auto expression = Expression::operation(
			SourceRange::unknown(),
			makeType<int>(),
			OperationKind::add,
			llvm::ArrayRef({
					Expression::constant(SourceRange::unknown(), makeType<int>(), 3),
					Expression::constant(SourceRange::unknown(), makeType<int>(), -1)
			}));

	ConstantFolder folder;

	if (folder.run<Expression>(*expression))
		FAIL();

	EXPECT_TRUE(expression->isa<Constant>());
	EXPECT_EQ(expression->get<Constant>()->get<BuiltInType::Integer>(), 2);
}

TEST(folderTest, sumInSubscriptionShouldFold)
{
	auto access = Expression::operation(
			SourceRange::unknown(),
			makeType<int>(),
			OperationKind::add,
			llvm::ArrayRef({
					Expression::constant(SourceRange::unknown(), makeType<int>(), 3),
					Expression::constant(SourceRange::unknown(), makeType<int>(), -1)
			}));

	auto subscription = Expression::operation(
			SourceRange::unknown(),
			makeType<int>(),
			OperationKind::subscription,
			llvm::ArrayRef({
					Expression::reference(SourceRange::unknown(), makeType<int>(10), "name"),
					std::move(access)
			}));

	ConstantFolder folder;
	auto& symbolTable = folder.getSymbolTable();
	ConstantFolder::SymbolTableScope scope(symbolTable);

	auto member = Member::build(SourceRange::unknown(), "name", makeType<int>(10), TypePrefix::none());
	symbolTable.insert(member->getName(), Symbol(*member));

	if (folder.run<Expression>(*subscription))
		FAIL();

	EXPECT_TRUE(subscription->isa<Operation>());
	auto* accessIndex = subscription->get<Operation>()->getArg(1);
	EXPECT_TRUE(accessIndex->isa<Constant>());
	EXPECT_EQ(accessIndex->get<Constant>()->get<BuiltInType::Integer>(), 2);
}

TEST(folderTest, sumInSubscriptionInDerShouldFold)
{
	auto access = Expression::operation(
			SourceRange::unknown(),
			makeType<int>(),
			OperationKind::add,
			llvm::ArrayRef({
					Expression::constant(SourceRange::unknown(), makeType<int>(), 3),
					Expression::constant(SourceRange::unknown(), makeType<int>(), -1)
			}));

	auto subscription = Expression::operation(
			SourceRange::unknown(),
			makeType<int>(),
			OperationKind::subscription,
			llvm::ArrayRef({
					Expression::reference(SourceRange::unknown(), makeType<int>(10), "name"),
					std::move(access)
			}));

	auto refToDer = Expression::reference(SourceRange::unknown(), Type::unknown(), "der");

	auto call = Expression::call(
			SourceRange::unknown(), Type::unknown(),
			move(refToDer), std::move(subscription));

	ConstantFolder folder;
	auto& symbolTable = folder.getSymbolTable();
	ConstantFolder::SymbolTableScope scope(symbolTable);

	auto member = Member::build(SourceRange::unknown(), "name", makeType<int>(10), TypePrefix::none());
	symbolTable.insert(member->getName(), Symbol(*member));

	if (folder.run<Expression>(*call))
		FAIL();

	EXPECT_TRUE(call->isa<Call>());
	auto* arg = call->get<Call>()->getArg(0);
	EXPECT_TRUE(arg->isa<Operation>());
	auto* accessIndex = arg->get<Operation>()->getArg(1);
	EXPECT_EQ(accessIndex->get<Constant>()->get<BuiltInType::Integer>(), 2);
}

TEST(folderTest, startDeclarationWithReference)	 // NOLINT
{
	Parser parser("model C"
								"  parameter Real A = 315.15;"
								"  Real[10, 10, 4] T(start = A);"
								"end C;");

	auto ast = parser.classDefinition();

	if (!ast || !*ast)
		FAIL();

	ConstantFolder folder;

	if (folder.run<Class>(**ast))
		FAIL();

	const auto* model = (*ast)->get<Model>();
	bool isConstant = model->getMembers()[1]->getStartOverload()->isa<Constant>();
	EXPECT_TRUE(isConstant);
}
