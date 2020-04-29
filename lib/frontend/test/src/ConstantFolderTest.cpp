#include "gtest/gtest.h"

#include "modelica/frontend/Constant.hpp"
#include "modelica/frontend/ConstantFolder.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/SymbolTable.hpp"
#include "modelica/frontend/Type.hpp"

using namespace std;
using namespace modelica;

TEST(folderTest, sumShouldFold)
{
	Expression exp = Expression::op<OperationKind::add>(
			makeType<int>(),
			Expression(makeType<int>(), 3),
			Expression(makeType<int>(), 4));
	ConstantFolder folder;
	if (folder.foldExpression(exp, SymbolTable()))
		FAIL();

	EXPECT_TRUE(exp.isA<Constant>());
	EXPECT_EQ(exp.getConstant().get<int>(), 7);
}

TEST(folderTest, subShouldFold)
{
	Expression exp = Expression::op<OperationKind::subtract>(
			makeType<int>(),
			Expression(makeType<int>(), 3),
			Expression(makeType<int>(), 2));
	ConstantFolder folder;
	if (folder.foldExpression(exp, SymbolTable()))
		FAIL();

	EXPECT_TRUE(exp.isA<Constant>());
	EXPECT_EQ(exp.getConstant().get<int>(), 1);
}
