#include "gtest/gtest.h"
#include "marco/AST/AST.h"

using namespace marco;
using namespace marco::ast;

TEST(AST, singleElementTupleCanBeBuilt)	 // NOLINT
{
	SourceRange location = SourceRange::unknown();
	auto reference = Expression::reference(location, makeType<int>(), "x");
	auto tuple = Expression::tuple(location, Type::unknown(), std::move(reference));
	EXPECT_EQ(tuple->get<Tuple>()->size(), 1);
}

TEST(AST, multipleElementsTupleCanBeBuilt)	// NOLINT
{
	SourceRange location = SourceRange::unknown();

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
