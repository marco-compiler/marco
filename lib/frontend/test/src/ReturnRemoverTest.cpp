#include <gtest/gtest.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/Parser.h>
#include <modelica/frontend/Passes.h>

using namespace modelica;
using namespace frontend;

TEST(ReturnRemover, returnInIf)	 // NOLINT
{
	Parser parser("function Foo"
								"  input Integer x;"
								"  output Integer y;"
								"algorithm"
								"  y := 0;"
								"  if x == 5 then"
								"    return;"
								"  end if;"
								"  y := x;"
								"  y := y * 2;"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	ReturnRemover pass;
	EXPECT_TRUE(!pass.run(*ast));

	auto& algorithm = *(*ast)->get<StandardFunction>()->getAlgorithms()[0];
	EXPECT_EQ(algorithm.size(), 3);
	EXPECT_TRUE(algorithm[2]->isa<IfStatement>());
	EXPECT_EQ((*algorithm[2]->get<IfStatement>())[0].size(), 2);
}

TEST(ReturnRemover, returnInWhile)	 // NOLINT
{
	Parser parser("function Foo"
								"  input Integer x;"
								"  output Integer y;"
								"algorithm"
								"  y := 0;"
								"  while y < 10 loop"
								"    if y == 5 then"
								"      return;"
								"    end if;"
								"    y := y + 1;"
								"    y := y * 2;"
								"  end while;"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	ReturnRemover pass;
	EXPECT_TRUE(!pass.run(*ast));

	auto& algorithm = *(*ast)->get<StandardFunction>()->getAlgorithms()[0];
	auto* whileLoop = algorithm[1]->get<WhileStatement>();
	EXPECT_EQ(whileLoop->size(), 2);
	EXPECT_TRUE((*whileLoop)[0]->isa<IfStatement>());
	EXPECT_EQ((*whileLoop)[1]->get<IfStatement>()->getBlock(0).size(), 2);
}

TEST(ReturnRemover, returnInNestedWhile)	 // NOLINT
{
	Parser parser("function Foo"
								"  input Integer x;"
								"  output Integer y;"
								"algorithm"
								"  y := 0;"
								"  while y < 10 loop"
								"    while y < 5 loop"
								"      return;"
								"    end while;"
								"    y := y + 1;"
								"    y := y * 2;"
								"  end while;"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	ReturnRemover pass;
	EXPECT_TRUE(!pass.run(*ast));

	auto& algorithm = *(*ast)->get<StandardFunction>()->getAlgorithms()[0];

	// Differently from the break case, where the inner loop could not break
	// the outer one, here the inner loop may stop the outer loop execution.
	auto* outerLoop = algorithm[1]->get<WhileStatement>();
	EXPECT_EQ(outerLoop->size(), 2);
}

TEST(ReturnRemover, returnInFor)	 // NOLINT
{
	Parser parser("function Foo"
								"  input Integer x;"
								"  output Integer y;"
								"algorithm"
								"  y := 0;"
								"  for i in 1:10 loop"
								"    if y < 5 then"
								"      return;"
								"    end if;"
								"    y := 1;"
								"    y := y + i;"
								"  end for;"
								"  y := y + 1;"
								"  y := y * 2;"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	ReturnRemover pass;
	EXPECT_TRUE(!pass.run(*ast));

	auto& algorithm = *(*ast)->get<StandardFunction>()->getAlgorithms()[0];
	EXPECT_EQ(algorithm.size(), 3);

	auto* forLoop = algorithm[1]->get<ForStatement>();
	EXPECT_EQ(forLoop->size(), 2);
}
