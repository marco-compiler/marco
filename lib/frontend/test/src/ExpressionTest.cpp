#include "gtest/gtest.h"

#include "modelica/frontend/Constant.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/Type.hpp"

using namespace std;
using namespace modelica;

TEST(expressionTest, constantsCanBeBuilt)
{
	Expression exp(makeType<int>(), 3);
	EXPECT_TRUE(exp.isA<Constant>());
	EXPECT_TRUE(exp.getConstant().isA<BuiltinType::Integer>());
	EXPECT_EQ(exp.getConstant().get<BuiltinType::Integer>(), 3);
	EXPECT_EQ(exp.getType().getBuiltIn(), BuiltinType::Integer);
	EXPECT_EQ(exp.getType().size(), 1);
}

TEST(expressionTest, operationsCanBeBuilt)
{
	Expression constant(makeType<int>(), 3);
	Expression exp =
			Expression::op<OperationKind::add>(makeType<int>(), constant, constant);
	EXPECT_TRUE(exp.isOperation());
	EXPECT_EQ(exp.getOperation().getKind(), OperationKind::add);
}
