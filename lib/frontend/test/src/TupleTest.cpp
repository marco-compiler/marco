#include <gtest/gtest.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/Parser.hpp>

using namespace modelica;
using namespace frontend;

TEST(AST, singleElementTupleCanBeBuilt)	 // NOLINT
{
	SourcePosition location = SourcePosition::unknown();
	auto reference = Expression::reference(location, makeType<int>(), "x");
	auto tuple = Expression::tuple(location, Type::unknown(), std::move(reference));
	EXPECT_EQ(tuple->get<Tuple>()->size(), 1);
}

TEST(AST, multipleElementsTupleCanBeBuilt)	// NOLINT
{
	SourcePosition location = SourcePosition::unknown();

	auto exp1 = Expression::reference(location, makeType<int>(), "x");
	auto exp2 = Expression::reference(location, makeType<float>(), "y");
	auto exp3 = Expression::reference(location, Type::unknown(), "z");

	auto tuple = Expression::tuple(
			location, Type::unknown(),
			llvm::ArrayRef({
					std::move(exp1),
					std::move(exp2),
					std::move(exp3)
			}));

	EXPECT_EQ(tuple->get<Tuple>()->size(), 3);
}

TEST(Parser, emptyTuple)	 // NOLINT
{
	Parser parser("() := Foo(time);");

	auto ast = parser.statement();
	ASSERT_FALSE(!ast);

	EXPECT_EQ((*ast)->get<AssignmentStatement>()->getDestinations()->get<Tuple>()->size(), 0);
}
