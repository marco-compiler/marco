#include "gtest/gtest.h"
#include "marco/utils/Shape.h"
#include "marco/utils/IRange.h"
#include <vector>

using namespace marco;
using namespace std;

using Indexes = std::vector<std::vector<long>>;

TEST(ShapeTest, Equality)
{
    EXPECT_EQ( Shape({}), Shape({}));

    EXPECT_EQ( Shape({1}), Shape({1}));
    EXPECT_EQ( Shape({2}), Shape({2}));

    EXPECT_EQ( Shape({2,3}), Shape({2,3}));
    EXPECT_EQ( Shape({2,3,4}), Shape({2,3,4}));

    EXPECT_EQ( Shape({2,{2,3}}), Shape({2,{2,3}}));
    EXPECT_EQ( Shape({2,{3,2},{{2,4,2},3}}), Shape({2,{3,2},{{2,4,2},3}}));

    // EXPECT_EQ( Shape({2,{3,2},{2,3}}), Shape({2,{3,2},{{2,2,2},{3,3,3}}}));  //is it needed? canonization?
}

TEST(ShapeTest, NonRagged)
{

    EXPECT_EQ( generateAllIndexes(Shape({})), Indexes({{}}) );

    EXPECT_EQ( generateAllIndexes(Shape({2})), Indexes({{0},{1}}) );
    
	EXPECT_EQ( generateAllIndexes(Shape({2,3})), Indexes({{0,0},{0,1},{0,2},
                                                          {1,0},{1,1},{1,2}}) );
}

TEST(ShapeTest, Ragged)
{
    Shape shape = {2,{2,3}};

	EXPECT_EQ( generateAllIndexes(shape), Indexes({{0,0},{0,1},
                                                   {1,0},{1,1},{1,2}}) );
}