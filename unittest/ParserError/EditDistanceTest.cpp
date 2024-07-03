#include "gtest/gtest.h"
#include "marco/Codegen/Lowering/Distance/EditDistance.h"

using namespace ::marco;
using namespace ::marco::codegen::lowering;

TEST(ParserError, deletion)
{
  std::string actual = "var";
  std::string expected = "var1";
  unsigned int distance = EditDistance::editDistance(actual, expected);
  ASSERT_EQ(distance, EditDistance::deletion_cost);
}

TEST(ParserError, transposition)
{
  std::string actual = "vra";
  std::string expected = "var";
  unsigned int distance = EditDistance::editDistance(actual, expected);
  ASSERT_EQ(distance, EditDistance::transposition_cost);
}

TEST(ParserError, substitution)
{
  std::string actual = "veriable";
  std::string expected = "variable";
  unsigned int distance = EditDistance::editDistance(actual, expected);
  unsigned int idx1 = EditDistance::charToIndex('a');
  unsigned int idx2 = EditDistance::charToIndex('e');
  ASSERT_EQ(distance, EditDistance::substitution_costs[idx1][idx2]);
}

TEST(ParserError, insertion)
{
  std::string actual = "variable1";
  std::string expected = "variable";
  unsigned int distance = EditDistance::editDistance(actual, expected);
  unsigned int idx1 = EditDistance::charToIndex('1');
  unsigned int idx2 = EditDistance::charToIndex('e');
  ASSERT_EQ(distance, EditDistance::base_insertion_cost + EditDistance::substitution_costs[idx1][idx2]);
}

TEST(ParserError, insertion_initial)
{
  std::string actual = "1_";
  std::string expected = "_";
  unsigned int distance = EditDistance::editDistance(actual, expected);
  ASSERT_EQ(distance, EditDistance::base_insertion_cost + EditDistance::largest_substitution_cost);
}

TEST(ParserError, deletion_initial)
{
  std::string actual = "ariable";
  std::string expected = "variable";
  unsigned int distance = EditDistance::editDistance(actual, expected);
  ASSERT_EQ(distance, EditDistance::deletion_cost);
}

TEST(ParserError, empty_string_actual)
{
  std::string actual = "";
  std::string expected = "var";
  unsigned int distance = EditDistance::editDistance(actual, expected);
  ASSERT_EQ(distance, 3*EditDistance::deletion_cost);
}

TEST(ParserError, empty_string_expected)
{
  std::string actual = "var";
  std::string expected = "";
  unsigned int distance = EditDistance::editDistance(actual, expected);
  ASSERT_EQ(distance, 3*(EditDistance::base_insertion_cost + EditDistance::largest_substitution_cost));
}