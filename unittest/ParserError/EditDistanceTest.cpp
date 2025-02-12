#include "marco/Codegen/Lowering/Distance/EditDistance.h"
#include "marco/Codegen/Lowering/Distance/EditDistanceParameters.h"
#include "gtest/gtest.h"

using namespace ::marco;
using namespace ::marco::codegen::lowering;

TEST(ParserError, deletion) {
  std::string actual = "var";
  std::string expected = "var1";
  unsigned int distance = editDistance(actual, expected);
  ASSERT_EQ(distance, editDistanceDeletionCost);
}

TEST(ParserError, transposition) {
  std::string actual = "vra";
  std::string expected = "var";
  unsigned int distance = editDistance(actual, expected);
  ASSERT_EQ(distance, editDistanceTranspositionCost);
}

TEST(ParserError, substitution) {
  std::string actual = "veriable";
  std::string expected = "variable";
  unsigned int distance = editDistance(actual, expected);
  unsigned int idx1 = charToSubstitutionCostsIndex('a');
  unsigned int idx2 = charToSubstitutionCostsIndex('e');
  ASSERT_EQ(distance, editDistanceSubsitutionCosts[idx1][idx2]);
}

TEST(ParserError, insertion) {
  std::string actual = "variable1";
  std::string expected = "variable";
  unsigned int distance = editDistance(actual, expected);
  unsigned int idx1 = charToSubstitutionCostsIndex('1');
  unsigned int idx2 = charToSubstitutionCostsIndex('e');
  ASSERT_EQ(distance, editDistanceBaseInsertionCost +
                          editDistanceSubsitutionCosts[idx1][idx2]);
}

TEST(ParserError, insertion_initial) {
  std::string actual = "1_";
  std::string expected = "_";
  unsigned int distance = editDistance(actual, expected);
  ASSERT_EQ(distance, editDistanceBaseInsertionCost +
                          editDistanceLargestSubstitutionCost);
}

TEST(ParserError, deletion_initial) {
  std::string actual = "ariable";
  std::string expected = "variable";
  unsigned int distance = editDistance(actual, expected);
  ASSERT_EQ(distance, editDistanceDeletionCost);
}

TEST(ParserError, empty_string_actual) {
  std::string actual = "";
  std::string expected = "var";
  unsigned int distance = editDistance(actual, expected);
  ASSERT_EQ(distance, 3 * editDistanceDeletionCost);
}

TEST(ParserError, empty_string_expected) {
  std::string actual = "var";
  std::string expected = "";
  unsigned int distance = editDistance(actual, expected);
  ASSERT_EQ(distance, 3 * (editDistanceBaseInsertionCost +
                           editDistanceLargestSubstitutionCost));
}