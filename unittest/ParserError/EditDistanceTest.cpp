#include "gtest/gtest.h"
#include "marco/Codegen/Lowering/Distance/EditDistance.h"

using namespace ::marco;
using namespace ::marco::codegen::lowering;

TEST(ParserError, deletion)
{
  const std::string actual = "var";
  const std::string expected = "var1";
  const unsigned int distance = EditDistance::editDistance(actual, expected);
  ASSERT_EQ(distance, EditDistance::deletion_cost);
}

TEST(ParserError, transposition)
{
  const std::string actual = "vra";
  const std::string expected = "var";
  const unsigned int distance = EditDistance::editDistance(actual, expected);
  ASSERT_EQ(distance, EditDistance::transposition_cost);
}

TEST(ParserError, substitution)
{
  const std::string actual = "veriable";
  const std::string expected = "variable";
  const unsigned int distance = EditDistance::editDistance(actual, expected);
  const unsigned int idx1 = EditDistance::charToIndex('a');
  const unsigned int idx2 = EditDistance::charToIndex('e');
  ASSERT_EQ(distance, EditDistance::substitution_costs[idx1][idx2]);
}

TEST(ParserError, insertion)
{
  const std::string actual = "variable1";
  const std::string expected = "variable";
  const unsigned int distance = EditDistance::editDistance(actual, expected);
  const unsigned int idx1 = EditDistance::charToIndex('1');
  const unsigned int idx2 = EditDistance::charToIndex('e');
  ASSERT_EQ(distance, EditDistance::base_insertion_cost + EditDistance::substitution_costs[idx1][idx2]);
}

TEST(ParserError, insertion_initial)
{
  const std::string actual = "1_";
  const std::string expected = "_";
  const unsigned int distance = EditDistance::editDistance(actual, expected);
  ASSERT_EQ(distance, EditDistance::base_insertion_cost + EditDistance::largest_substitution_cost);
}

TEST(ParserError, deletion_initial)
{
  const std::string actual = "ariable";
  const std::string expected = "variable";
  const unsigned int distance = EditDistance::editDistance(actual, expected);
  ASSERT_EQ(distance, EditDistance::deletion_cost);
}

TEST(ParserError, empty_string_actual)
{
  const std::string actual = "";
  const std::string expected = "var";
  const unsigned int distance = EditDistance::editDistance(actual, expected);
  ASSERT_EQ(distance, 3*EditDistance::deletion_cost);
}

TEST(ParserError, empty_string_expected)
{
  const std::string actual = "var";
  const std::string expected = "";
  const unsigned int distance = EditDistance::editDistance(actual, expected);
  ASSERT_EQ(distance, 3*(EditDistance::base_insertion_cost + EditDistance::largest_substitution_cost));
}