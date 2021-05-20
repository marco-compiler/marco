#include <gtest/gtest.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/Parser.h>
#include <modelica/frontend/Passes.h>

using namespace modelica;
using namespace frontend;

TEST(BreakRemover, breakInWhile)	 // NOLINT
{
	Parser parser("function Foo"
								"  input Integer x;"
								"  output Integer y;"
								"algorithm"
								"  y := 0;"
								"  while y < 10 loop"
								"    if y == 5 then"
								"      break;"
								"    end if;"
								"    y := y + 1;"
								"    y := y * 2;"
								"  end while;"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	BreakRemover pass;
	EXPECT_TRUE(!pass.run(**ast));

	auto& algorithm = *(*ast)->get<StandardFunction>()->getAlgorithms()[0];
	auto* whileLoop = algorithm[1]->get<WhileStatement>();
	EXPECT_EQ(whileLoop->size(), 2);
	EXPECT_TRUE((*whileLoop)[1]->isa<IfStatement>());
	EXPECT_EQ((*whileLoop)[1]->get<IfStatement>()->getBlock(0).size(), 2);
}

TEST(BreakRemover, breakInNestedWhile)	 // NOLINT
{
	Parser parser("function Foo"
								"  input Integer x;"
								"  output Integer y;"
								"algorithm"
								"  y := 0;"
								"  while y < 10 loop"
								"    while y < 5 loop"
								"      break;"
								"    end while;"
								"    y := y + 1;"
								"    y := y * 2;"
								"  end while;"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	BreakRemover pass;
	EXPECT_TRUE(!pass.run(**ast));

	auto& algorithm = *(*ast)->get<StandardFunction>()->getAlgorithms()[0];

	// The outer loop should be unchanged, because the inner loop can't break
	// the outer one.
	auto* outerLoop = algorithm[1]->get<WhileStatement>();
	EXPECT_EQ(outerLoop->size(), 3);

	// Inside the inner loop, there should be just an assignment to the break
	// check variable, because in the original body there are no statements
	// that can be avoided.
	auto* innerLoop = (*outerLoop)[0]->get<WhileStatement>();
	EXPECT_EQ(innerLoop->size(), 1);
	EXPECT_TRUE((*innerLoop)[0]->isa<AssignmentStatement>());
	EXPECT_EQ((*innerLoop)[0]->get<AssignmentStatement>()->getDestinations()->get<Tuple>()->getArg(0)->get<ReferenceAccess>()->getName(), "__mustBreak2");
}

TEST(BreakRemover, breakInFor)	 // NOLINT
{
	Parser parser("function Foo"
								"  input Integer x;"
								"  output Integer y;"
								"algorithm"
								"  y := 0;"
								"  for i in 1:10 loop"
								"    if y == 5 then"
								"      break;"
								"    end if;"
								"    y := y + 1;"
								"    y := y * 2;"
								"  end for;"
								"end Foo;");

	auto ast = parser.classDefinition();
	ASSERT_FALSE(!ast);

	BreakRemover pass;
	EXPECT_TRUE(!pass.run(**ast));

	auto& algorithm = *(*ast)->get<StandardFunction>()->getAlgorithms()[0];
	auto* forLoop = algorithm[1]->get<ForStatement>();
	EXPECT_EQ(forLoop->size(), 2);
	EXPECT_TRUE((*forLoop)[1]->isa<IfStatement>());
	EXPECT_EQ((*forLoop)[1]->get<IfStatement>()->getBlock(0).size(), 2);
}
