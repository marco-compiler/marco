#include "gtest/gtest.h"

#include "modelica/frontend/Constant.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/Type.hpp"

using namespace modelica;
using namespace std;

TEST(expressionTest, constantsCanBeBuilt)
{
	SourcePosition location("-", 0, 0);
	Expression exp(location, makeType<int>(), 3);
	EXPECT_TRUE(exp.isA<Constant>());
	EXPECT_TRUE(exp.get<Constant>().isA<BuiltInType::Integer>());
	EXPECT_EQ(exp.get<Constant>().get<BuiltInType::Integer>(), 3);
	EXPECT_EQ(exp.getType().get<BuiltInType>(), BuiltInType::Integer);
	EXPECT_EQ(exp.getType().size(), 1);
}

TEST(expressionTest, operationsCanBeBuilt)
{
	SourcePosition location("-", 0, 0);
	Expression constant(location, makeType<int>(), 3);
	Expression exp =
			Expression::op<OperationKind::add>(location, makeType<int>(), constant, constant);
	EXPECT_TRUE(exp.isA<Operation>());
	EXPECT_EQ(exp.get<Operation>().getKind(), OperationKind::add);
}
