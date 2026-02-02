#include "marco/Modeling/RTree.h"
#include "llvm/Support/raw_os_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace ::marco::modeling;

namespace {
class Object {
  MultidimensionalRange range;

public:
  Object(MultidimensionalRange range) : range(std::move(range)) {}

  const MultidimensionalRange &operator*() const { return range; }

  bool operator==(const Object &other) const { return range == other.range; }
};

std::ostream &operator<<(std::ostream &os, const Object &obj) {
  llvm::raw_os_ostream ros(os);
  ros << *obj;
  ros.flush();
  return os;
}
} // namespace

namespace llvm {
template <>
struct DenseMapInfo<Object> {
  static inline Object getEmptyKey() {
    return Object(DenseMapInfo<MultidimensionalRange>::getEmptyKey());
  }

  static inline Object getTombstoneKey() {
    return Object(DenseMapInfo<MultidimensionalRange>::getTombstoneKey());
  }

  static unsigned getHashValue(const Object &obj) { return hash_value(*obj); }

  static bool isEqual(const Object &lhs, const Object &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace marco::modeling {
template <>
struct RTreeInfo<Object> {
  static const MultidimensionalRange &getShape(const Object &obj) {
    return *obj;
  }

  static bool isEqual(const Object &first, const Object &second) {
    return *first == *second;
  }
};
} // namespace marco::modeling

TEST(RTree, empty) {
  RTree<Object> emptyTree;
  EXPECT_TRUE(emptyTree.empty());

  RTree<Object> nonEmptyTree;
  nonEmptyTree.insert(Object(MultidimensionalRange(Range(2, 5))));
  EXPECT_FALSE(nonEmptyTree.empty());
}

TEST(RTree, clear) {
  RTree<Object> tree;
  tree.insert(Object(MultidimensionalRange(Range(2, 5))));
  ASSERT_FALSE(tree.empty());

  tree.clear();
  EXPECT_TRUE(tree.empty());
}

TEST(RTree, containsObject) {
  RTree<Object> tree;
  tree.insert(Object(MultidimensionalRange(Range(2, 5))));
  EXPECT_TRUE(tree.contains(Object(MultidimensionalRange(Range(2, 5)))));
  EXPECT_FALSE(tree.contains(Object(MultidimensionalRange(Range(3, 4)))));
}

TEST(RTree, insertObject) {
  llvm::SmallVector<Object> objects;
  objects.emplace_back(MultidimensionalRange(Range(8, 9)));
  objects.emplace_back(MultidimensionalRange(Range(4, 6)));
  objects.emplace_back(MultidimensionalRange(Range(7, 8)));
  objects.emplace_back(MultidimensionalRange(Range(0, 1)));
  objects.emplace_back(MultidimensionalRange(Range(5, 8)));
  objects.emplace_back(MultidimensionalRange(Range(4, 8)));
  objects.emplace_back(MultidimensionalRange(Range(3, 7)));
  objects.emplace_back(MultidimensionalRange(Range(0, 3)));
  objects.emplace_back(MultidimensionalRange(Range(4, 5)));
  objects.emplace_back(MultidimensionalRange(Range(2, 8)));
  objects.emplace_back(MultidimensionalRange(Range(3, 8)));
  objects.emplace_back(MultidimensionalRange(Range(4, 7)));
  objects.emplace_back(MultidimensionalRange(Range(2, 9)));
  objects.emplace_back(MultidimensionalRange(Range(0, 5)));
  objects.emplace_back(MultidimensionalRange(Range(7, 9)));
  objects.emplace_back(MultidimensionalRange(Range(1, 2)));
  objects.emplace_back(MultidimensionalRange(Range(1, 6)));
  objects.emplace_back(MultidimensionalRange(Range(2, 5)));
  objects.emplace_back(MultidimensionalRange(Range(6, 7)));
  objects.emplace_back(MultidimensionalRange(Range(0, 7)));
  objects.emplace_back(MultidimensionalRange(Range(3, 5)));
  objects.emplace_back(MultidimensionalRange(Range(3, 4)));

  RTree<Object> tree;

  for (const Object &object : objects) {
    tree.insert(object);
  }

  llvm::SmallVector<Object> actualObjects(tree.begin(), tree.end());

  EXPECT_THAT(actualObjects, testing::UnorderedElementsAreArray(objects.begin(),
                                                                objects.end()));
}

TEST(RTree, insertTree) {
  llvm::SmallVector<Object> firstObjects;
  firstObjects.emplace_back(MultidimensionalRange(Range(8, 9)));
  firstObjects.emplace_back(MultidimensionalRange(Range(4, 6)));
  firstObjects.emplace_back(MultidimensionalRange(Range(7, 8)));
  firstObjects.emplace_back(MultidimensionalRange(Range(0, 1)));
  firstObjects.emplace_back(MultidimensionalRange(Range(5, 8)));

  llvm::SmallVector<Object> secondObjects;
  secondObjects.emplace_back(MultidimensionalRange(Range(4, 8)));
  secondObjects.emplace_back(MultidimensionalRange(Range(3, 7)));
  secondObjects.emplace_back(MultidimensionalRange(Range(0, 3)));
  secondObjects.emplace_back(MultidimensionalRange(Range(4, 5)));
  secondObjects.emplace_back(MultidimensionalRange(Range(2, 8)));
  secondObjects.emplace_back(MultidimensionalRange(Range(3, 8)));
  secondObjects.emplace_back(MultidimensionalRange(Range(4, 7)));
  secondObjects.emplace_back(MultidimensionalRange(Range(2, 9)));
  secondObjects.emplace_back(MultidimensionalRange(Range(0, 5)));
  secondObjects.emplace_back(MultidimensionalRange(Range(7, 9)));
  secondObjects.emplace_back(MultidimensionalRange(Range(1, 2)));
  secondObjects.emplace_back(MultidimensionalRange(Range(1, 6)));
  secondObjects.emplace_back(MultidimensionalRange(Range(2, 5)));
  secondObjects.emplace_back(MultidimensionalRange(Range(6, 7)));
  secondObjects.emplace_back(MultidimensionalRange(Range(0, 7)));
  secondObjects.emplace_back(MultidimensionalRange(Range(3, 5)));
  secondObjects.emplace_back(MultidimensionalRange(Range(3, 4)));

  RTree<Object> firstTree;
  RTree<Object> secondTree;

  for (const Object &object : firstObjects) {
    firstTree.insert(object);
  }

  for (const Object &object : secondObjects) {
    secondTree.insert(object);
  }

  firstTree.insert(secondTree);

  llvm::SmallVector<Object> actualObjects(firstTree.begin(), firstTree.end());
  llvm::SmallVector<Object> expectedObjects(firstObjects);
  llvm::append_range(expectedObjects, secondObjects);

  EXPECT_THAT(actualObjects,
              testing::UnorderedElementsAreArray(expectedObjects));
}

TEST(RTree, removeObject) {
  llvm::SmallVector<Object> objects;
  objects.emplace_back(MultidimensionalRange(Range(8, 9)));
  objects.emplace_back(MultidimensionalRange(Range(4, 6)));
  objects.emplace_back(MultidimensionalRange(Range(7, 8)));
  objects.emplace_back(MultidimensionalRange(Range(0, 1)));
  objects.emplace_back(MultidimensionalRange(Range(5, 8)));
  objects.emplace_back(MultidimensionalRange(Range(4, 8)));
  objects.emplace_back(MultidimensionalRange(Range(3, 7)));
  objects.emplace_back(MultidimensionalRange(Range(0, 3)));
  objects.emplace_back(MultidimensionalRange(Range(4, 5)));
  objects.emplace_back(MultidimensionalRange(Range(2, 8)));
  objects.emplace_back(MultidimensionalRange(Range(3, 8)));
  objects.emplace_back(MultidimensionalRange(Range(4, 7)));
  objects.emplace_back(MultidimensionalRange(Range(2, 9)));
  objects.emplace_back(MultidimensionalRange(Range(0, 5)));
  objects.emplace_back(MultidimensionalRange(Range(7, 9)));
  objects.emplace_back(MultidimensionalRange(Range(1, 2)));
  objects.emplace_back(MultidimensionalRange(Range(1, 6)));
  objects.emplace_back(MultidimensionalRange(Range(2, 5)));
  objects.emplace_back(MultidimensionalRange(Range(6, 7)));
  objects.emplace_back(MultidimensionalRange(Range(0, 7)));
  objects.emplace_back(MultidimensionalRange(Range(3, 5)));
  objects.emplace_back(MultidimensionalRange(Range(3, 4)));

  RTree<Object> tree;

  for (const Object &object : objects) {
    tree.insert(object);
  }

  size_t removedIndex = 2;
  tree.remove(objects[removedIndex]);
  llvm::SmallVector<Object> expectedObjects;

  for (size_t i = 0; i < objects.size(); ++i) {
    if (i != removedIndex) {
      expectedObjects.emplace_back(objects[i]);
    }
  }

  llvm::SmallVector<Object> actualObjects(tree.begin(), tree.end());

  EXPECT_THAT(actualObjects,
              testing::UnorderedElementsAreArray(expectedObjects));
}

TEST(RTree, removeTree) {
  llvm::SmallVector<Object> firstObjects;
  firstObjects.emplace_back(MultidimensionalRange(Range(8, 9)));
  firstObjects.emplace_back(MultidimensionalRange(Range(4, 6)));
  firstObjects.emplace_back(MultidimensionalRange(Range(7, 8)));
  firstObjects.emplace_back(MultidimensionalRange(Range(0, 1)));
  firstObjects.emplace_back(MultidimensionalRange(Range(5, 8)));
  firstObjects.emplace_back(MultidimensionalRange(Range(4, 8)));
  firstObjects.emplace_back(MultidimensionalRange(Range(3, 7)));
  firstObjects.emplace_back(MultidimensionalRange(Range(0, 3)));
  firstObjects.emplace_back(MultidimensionalRange(Range(4, 5)));
  firstObjects.emplace_back(MultidimensionalRange(Range(2, 8)));
  firstObjects.emplace_back(MultidimensionalRange(Range(3, 8)));
  firstObjects.emplace_back(MultidimensionalRange(Range(4, 7)));
  firstObjects.emplace_back(MultidimensionalRange(Range(2, 9)));
  firstObjects.emplace_back(MultidimensionalRange(Range(0, 5)));
  firstObjects.emplace_back(MultidimensionalRange(Range(7, 9)));
  firstObjects.emplace_back(MultidimensionalRange(Range(1, 2)));
  firstObjects.emplace_back(MultidimensionalRange(Range(1, 6)));
  firstObjects.emplace_back(MultidimensionalRange(Range(2, 5)));
  firstObjects.emplace_back(MultidimensionalRange(Range(6, 7)));
  firstObjects.emplace_back(MultidimensionalRange(Range(0, 7)));
  firstObjects.emplace_back(MultidimensionalRange(Range(3, 5)));
  firstObjects.emplace_back(MultidimensionalRange(Range(3, 4)));

  RTree<Object> firstTree;

  for (const Object &object : firstObjects) {
    firstTree.insert(object);
  }

  RTree<Object> secondTree;

  for (size_t i = 2, e = firstObjects.size(); i < e; ++i) {
    secondTree.insert(firstObjects[i]);
  }

  firstTree.remove(secondTree);

  llvm::SmallVector<Object> expectedObjects(firstObjects.begin(),
                                            std::next(firstObjects.begin(), 2));

  llvm::SmallVector<Object> actualObjects(firstTree.begin(), firstTree.end());

  EXPECT_THAT(actualObjects,
              testing::UnorderedElementsAreArray(expectedObjects));
}

TEST(RTree, randomInsertionsAndRemovals) {
  RTree<Object> tree;
  llvm::SmallVector<Object> objects;

  llvm::SmallVector<int64_t> ranges(
      {78, 120, 40, 69,  35, 84,  32, 35,  27, 68,  7,  54,  63, 68,  97, 126,
       46, 79,  41, 65,  29, 35,  51, 69,  36, 76,  63, 92,  64, 102, 61, 107,
       28, 32,  23, 40,  51, 57,  20, 51,  48, 77,  26, 38,  84, 110, 42, 75,
       58, 92,  5,  43,  38, 45,  6,  31,  96, 116, 55, 68,  56, 75,  59, 97,
       21, 57,  5,  30,  42, 68,  54, 95,  5,  36,  2,  42,  8,  53,  23, 40,
       79, 110, 5,  25,  37, 49,  96, 132, 32, 34,  49, 91,  71, 81,  78, 121,
       96, 130, 16, 55,  60, 81,  29, 46,  2,  34,  7,  18,  27, 59,  28, 37,
       63, 99,  27, 30,  48, 74,  37, 69,  78, 118, 22, 72,  0,  3,   41, 88,
       87, 97,  34, 82,  31, 47,  13, 48,  98, 120, 96, 122, 4,  30,  35, 53,
       62, 78,  20, 32,  90, 100, 44, 66,  0,  19,  70, 71,  22, 37,  48, 58,
       73, 109, 56, 62,  50, 73,  91, 141, 95, 133, 26, 76,  64, 77,  18, 46,
       77, 118, 90, 110, 0,  37,  40, 41,  4,  18,  53, 82,  27, 31,  87, 88,
       88, 134, 57, 99,  17, 66,  40, 55,  88, 105, 63, 66,  80, 115, 79, 89,
       24, 46,  81, 108, 7,  29,  77, 91,  36, 69,  41, 55,  87, 116, 16, 45,
       76, 100, 19, 63,  74, 84,  58, 71,  77, 101, 66, 77,  7,  56,  19, 54,
       19, 22,  60, 88,  74, 114, 90, 101, 23, 56,  26, 38,  12, 55,  39, 78,
       67, 76,  34, 78,  69, 114, 7,  56,  17, 42,  8,  36,  72, 103, 13, 55,
       32, 56,  70, 79,  63, 75,  21, 60,  45, 93,  49, 57,  41, 82,  98, 109,
       98, 131, 54, 74,  76, 88,  18, 64,  87, 116, 74, 84,  10, 48,  51, 95,
       61, 83,  53, 80,  34, 59,  14, 46,  73, 90,  41, 58,  56, 96,  27, 35,
       23, 57,  26, 76,  46, 93,  96, 131, 27, 49,  93, 131, 58, 105, 80, 102,
       70, 105, 47, 52,  60, 75,  88, 125, 30, 60,  52, 89,  20, 50,  93, 137,
       64, 87,  94, 106, 68, 109, 45, 93,  13, 52,  85, 108, 35, 51,  95, 101,
       51, 95,  61, 76,  57, 107, 50, 90,  78, 81,  25, 26,  34, 56,  93, 142,
       43, 81,  9,  23,  30, 37,  11, 55,  45, 94,  67, 100, 65, 79,  37, 55,
       58, 59,  33, 51,  52, 86,  6,  37,  37, 71,  33, 57,  54, 81,  74, 124,
       66, 102, 65, 112, 42, 71,  91, 129, 76, 88,  19, 61,  26, 35,  10, 45,
       10, 54,  3,  16,  78, 88,  45, 64,  92, 121, 91, 141, 4,  22,  48, 71,
       53, 69,  70, 116, 93, 108, 34, 54,  25, 31,  11, 13,  65, 87,  87, 113,
       17, 60,  90, 136, 51, 87,  15, 61,  13, 23,  46, 66,  76, 124, 92, 122,
       12, 25,  76, 83,  76, 89,  75, 79,  67, 104, 56, 89,  60, 106, 60, 88,
       37, 38,  24, 65,  37, 78,  88, 89,  99, 134, 71, 99,  81, 97,  9,  54,
       30, 66,  52, 59,  48, 76,  12, 30,  66, 85,  52, 81,  16, 29,  55, 59,
       14, 46,  46, 48,  71, 106, 53, 76,  68, 93,  2,  5,   92, 104, 48, 71,
       48, 49,  80, 129, 79, 122, 16, 62,  13, 34,  23, 53,  84, 115, 34, 83,
       62, 93,  1,  37,  14, 19,  60, 96,  30, 43,  37, 60,  73, 109, 96, 120,
       85, 113, 22, 37,  69, 110, 62, 97,  10, 48,  15, 60,  20, 21,  44, 77,
       82, 128, 19, 69,  51, 83,  36, 70,  93, 117, 8,  27,  10, 15,  44, 92,
       81, 98,  63, 67,  8,  34,  37, 56,  15, 19,  64, 100, 55, 64,  69, 109,
       5,  44,  38, 45,  72, 97,  89, 105, 99, 147, 36, 46,  54, 85,  56, 92,
       98, 121, 38, 45,  97, 124, 76, 89,  81, 124, 99, 136, 0,  19,  77, 83,
       9,  26,  13, 45,  42, 47,  98, 143, 4,  41,  55, 64,  18, 33,  93, 110,
       36, 71,  74, 108, 60, 63,  98, 142, 94, 144, 81, 128, 18, 27,  3,  31,
       26, 43,  60, 80,  23, 34,  15, 43,  97, 118, 85, 101, 34, 65,  84, 105,
       14, 25,  56, 83,  13, 20,  19, 29,  55, 56,  6,  30,  59, 69,  52, 88,
       78, 91,  6,  8,   23, 45,  28, 51,  92, 108, 87, 116, 47, 71,  49, 62,
       34, 40,  40, 90,  11, 22,  10, 29,  12, 29,  92, 116, 78, 125, 61, 68,
       61, 81,  9,  44,  41, 81,  8,  44,  54, 102, 63, 65,  71, 86,  15, 23,
       69, 76,  8,  41,  68, 87,  51, 82,  87, 133, 56, 72,  91, 109, 73, 78,
       36, 71,  40, 70,  23, 72,  66, 96,  96, 127, 82, 100, 94, 143, 76, 93,
       6,  41,  0,  25,  54, 58,  6,  48,  50, 63,  58, 103, 31, 65,  98, 117,
       17, 57,  49, 90,  87, 104, 21, 57,  46, 51,  2,  45,  54, 85,  60, 71,
       66, 80,  86, 108, 16, 59,  14, 34,  7,  31,  13, 52,  56, 70,  58, 85,
       2,  11,  66, 109, 24, 65,  77, 100, 44, 77,  14, 63,  62, 90,  10, 42,
       40, 89,  2,  11,  92, 111, 79, 129, 41, 84,  90, 91,  8,  57,  76, 87,
       56, 101, 4,  35,  34, 67,  2,  33,  14, 34,  80, 109, 48, 89,  9,  48,
       90, 104, 48, 82,  32, 61,  34, 60,  72, 97,  75, 106, 73, 77,  43, 75,
       98, 146, 62, 97,  29, 46,  15, 61,  37, 83,  24, 60,  38, 74,  73, 102,
       49, 74,  11, 45,  2,  51,  58, 85,  22, 57,  7,  53,  89, 90,  77, 117,
       97, 139, 24, 54,  59, 99,  74, 122, 86, 87,  82, 107, 36, 45,  5,  43,
       34, 53,  70, 107, 66, 95,  14, 54,  14, 36,  36, 41,  23, 39,  43, 67,
       8,  28,  52, 71,  10, 39,  65, 113, 29, 79,  73, 91,  59, 88,  54, 98,
       47, 74,  81, 97,  56, 103, 54, 76,  69, 113, 27, 71,  8,  31,  66, 84,
       42, 63,  37, 40,  48, 51,  1,  31,  53, 79,  46, 62,  5,  8,   58, 61,
       28, 71,  70, 106, 40, 65,  8,  18,  19, 55,  54, 83,  9,  30,  97, 101,
       42, 77,  56, 98,  88, 98,  20, 64,  36, 56,  58, 101, 71, 91,  46, 47,
       61, 78,  85, 87,  93, 137, 62, 75,  80, 98,  92, 132, 37, 77,  93, 124,
       75, 77,  23, 39,  62, 108, 9,  59,  14, 34,  41, 78,  88, 126, 86, 88,
       4,  28,  2,  50,  68, 84,  61, 62,  82, 87,  41, 63,  95, 130, 3,  27,
       87, 114, 38, 39,  74, 124, 1,  40,  71, 114, 74, 84,  81, 94,  61, 97,
       35, 51,  34, 40,  32, 81,  55, 72,  2,  50,  38, 88,  83, 127, 72, 96,
       72, 86,  25, 72,  62, 89,  34, 68,  20, 30,  45, 47,  21, 30,  39, 49,
       73, 99,  64, 71,  73, 96,  72, 100, 21, 34,  77, 84,  56, 106, 79, 108,
       12, 19,  26, 54,  82, 93,  12, 17,  69, 77,  8,  52,  15, 63,  4,  46,
       24, 45,  99, 148, 92, 116, 25, 41,  86, 89,  22, 67,  4,  8,   74, 91,
       10, 11,  95, 140, 60, 69,  51, 84,  17, 27,  77, 113, 6,  38,  26, 57,
       1,  29,  80, 125, 0,  9,   11, 48,  60, 94,  32, 49,  39, 46,  33, 83,
       6,  37,  45, 65,  88, 135, 53, 62,  55, 86,  43, 57,  11, 33,  96, 109,
       98, 125, 58, 107, 34, 55,  37, 85,  55, 77,  63, 108, 28, 77,  95, 132,
       29, 71,  5,  25,  37, 48,  27, 73,  40, 63,  58, 60,  95, 100, 16, 60,
       31, 56,  44, 62,  96, 130, 64, 67,  54, 85,  98, 131, 28, 73,  21, 31,
       35, 64,  79, 104, 89, 98,  69, 99,  33, 62,  33, 62,  82, 132, 22, 38,
       75, 94,  83, 106, 51, 101, 76, 82,  79, 106, 40, 51,  20, 34,  69, 77,
       91, 92,  84, 117, 11, 15,  62, 107, 33, 79,  72, 91,  96, 143, 85, 107,
       64, 85,  95, 113, 20, 42,  73, 75,  50, 66,  11, 32,  30, 64,  30, 52,
       85, 100, 54, 101, 19, 38,  92, 98,  15, 31,  73, 87,  11, 20,  34, 63,
       79, 110, 95, 97,  53, 74,  54, 58,  37, 56,  26, 44,  51, 60,  41, 80,
       22, 70,  85, 129, 67, 97,  98, 131, 94, 118, 45, 54,  34, 66,  86, 102,
       13, 47,  16, 36,  54, 75,  74, 116, 88, 91,  11, 53,  60, 63,  80, 115,
       99, 117, 28, 45,  46, 73,  0,  44,  50, 98,  51, 88,  28, 68,  1,  46,
       22, 40,  65, 94,  39, 79,  72, 103, 92, 126, 71, 76,  35, 39,  39, 76,
       72, 90,  4,  26,  45, 50,  16, 64,  53, 73,  83, 117, 8,  43,  27, 60,
       53, 96,  11, 55,  84, 118, 25, 52,  66, 115, 32, 36,  4,  26,  91, 120,
       90, 136, 51, 88,  51, 69,  35, 40,  86, 106, 37, 84,  55, 71,  81, 91,
       9,  52,  54, 98,  27, 59,  21, 67,  79, 84,  98, 132, 77, 119, 12, 31,
       36, 50,  4,  42,  83, 125, 91, 111, 10, 39,  68, 87,  45, 95,  79, 87,
       93, 129, 0,  21,  16, 41,  65, 111, 80, 96,  79, 87,  56, 100, 25, 70,
       58, 90,  31, 73,  73, 98,  63, 99,  54, 86,  53, 56,  32, 65,  59, 85,
       17, 29,  45, 81,  37, 50,  81, 99,  29, 42,  75, 113, 55, 58,  33, 49,
       34, 49,  7,  17,  90, 111, 94, 139, 53, 54,  48, 84,  84, 92,  10, 15,
       71, 79,  39, 48,  21, 42,  26, 27,  32, 36,  39, 79,  5,  28,  5,  45,
       88, 103, 50, 81,  34, 80,  26, 64,  97, 123, 72, 104, 34, 69,  35, 41,
       41, 67,  14, 27,  95, 138, 64, 95,  45, 51,  71, 72,  79, 108, 42, 60,
       92, 135, 49, 76,  89, 115, 65, 102, 50, 90,  68, 104, 23, 27,  42, 59,
       30, 37,  78, 107, 48, 93,  10, 54,  1,  33,  96, 127, 10, 49,  99, 104,
       82, 131, 81, 104, 24, 73,  10, 37,  88, 117, 63, 77,  34, 41,  80, 97,
       14, 25,  94, 110, 57, 62,  60, 69,  86, 93,  39, 88,  46, 85,  2,  32,
       39, 75,  53, 67,  86, 100, 41, 68,  94, 100, 89, 120, 11, 33,  96, 124,
       32, 74,  92, 132, 95, 99,  47, 81,  9,  48,  33, 42,  77, 116, 37, 54,
       23, 64,  31, 41,  5,  28,  35, 85,  29, 57,  79, 122, 98, 125, 70, 101,
       19, 32,  71, 88,  67, 89,  52, 80,  9,  45,  85, 124, 73, 98,  6,  56,
       16, 56,  10, 32,  62, 111, 73, 117, 75, 78,  36, 60,  80, 87,  56, 58,
       20, 50,  18, 57,  50, 71,  17, 30,  7,  12,  50, 84,  28, 38,  82, 129,
       98, 141, 17, 30,  90, 133, 56, 74,  47, 90,  91, 119, 50, 100, 79, 100,
       78, 128, 10, 42,  71, 101, 93, 122, 33, 79,  63, 77,  4,  50,  9,  13,
       90, 120, 15, 48,  21, 43,  50, 69,  15, 59,  48, 64,  44, 74,  88, 111,
       28, 29,  55, 57,  30, 79,  81, 95,  44, 90,  29, 30,  42, 81,  3,  38,
       17, 39,  17, 57,  94, 114, 59, 70,  64, 72,  75, 84,  86, 102, 82, 97,
       66, 105, 65, 112, 38, 86,  11, 46,  44, 85,  35, 72,  31, 72,  71, 120,
       61, 102, 39, 46,  59, 109, 66, 90,  58, 102, 83, 129, 61, 77,  11, 39,
       3,  33,  75, 120, 76, 115, 28, 51,  31, 45,  58, 71,  54, 84,  62, 80,
       21, 24,  23, 54,  53, 95,  55, 67,  87, 126, 8,  9,   4,  25,  79, 89,
       49, 54,  53, 81,  44, 77,  49, 75,  97, 105, 89, 91,  89, 92,  71, 82,
       6,  53,  43, 55,  88, 137, 74, 102, 89, 123, 27, 73,  3,  10,  4,  7,
       12, 23,  29, 38,  44, 75,  36, 78,  37, 63,  45, 74,  29, 48,  41, 77});

  for (size_t i = 0; i < ranges.size(); i += 2) {
    MultidimensionalRange object(Range(ranges[i], ranges[i + 1]));
    tree.insert(object);
    objects.push_back(object);
  }

  llvm::SmallVector<Object> actualObjects(tree.begin(), tree.end());
  ASSERT_THAT(actualObjects, testing::UnorderedElementsAreArray(objects));

  llvm::SmallVector<size_t> removedIndices(
      {60, 88,  30, 64,  54, 93,  76, 100, 69, 109, 54, 84,  9,  52,  46, 62,
       8,  31,  59, 88,  70, 105, 42, 56,  90, 101, 63, 67,  79, 129, 37, 56,
       88, 131, 48, 93,  37, 63,  78, 128, 65, 71,  6,  37,  11, 13,  26, 54,
       41, 88,  50, 98,  87, 114, 21, 31,  57, 107, 68, 80,  53, 76,  14, 46,
       63, 108, 50, 73,  33, 62,  11, 15,  28, 77,  37, 83,  74, 122, 41, 81,
       63, 92,  30, 79,  55, 86,  56, 83,  52, 76,  88, 134, 58, 90,  62, 111,
       60, 75,  23, 57,  98, 121, 35, 39,  50, 69,  48, 76,  85, 100, 17, 57,
       47, 52,  75, 77,  71, 79,  16, 60,  21, 24,  18, 64,  46, 51,  65, 79,
       53, 95,  71, 76,  33, 49,  51, 55,  47, 74,  56, 70,  71, 101, 73, 91,
       12, 17,  61, 81,  26, 61,  83, 127, 61, 68,  9,  23,  2,  51,  10, 45,
       32, 81,  61, 97,  4,  30,  34, 67,  14, 25,  49, 90,  45, 90,  91, 119,
       50, 84,  68, 104, 89, 98,  1,  40,  27, 35,  46, 66,  81, 95,  19, 22,
       25, 31,  94, 139, 12, 60,  73, 98,  34, 78,  34, 80,  23, 40,  95, 130,
       46, 47,  59, 85,  2,  11,  77, 117, 5,  8,   20, 49,  5,  45,  45, 95,
       34, 56,  51, 83,  64, 95,  73, 78,  46, 85,  39, 76,  9,  48,  10, 49,
       0,  3,   2,  50,  36, 78,  82, 132, 29, 30,  8,  43,  23, 33,  86, 88,
       23, 53,  58, 61,  33, 57,  73, 109, 88, 105, 51, 82,  48, 71,  0,  9,
       16, 41,  16, 29,  94, 100, 72, 99,  43, 67,  20, 30,  1,  37,  23, 39,
       37, 77,  89, 91,  3,  10,  12, 31,  80, 125, 42, 59,  15, 59,  37, 69,
       85, 108, 85, 124, 29, 42,  46, 81,  7,  56,  67, 85,  75, 84,  56, 98,
       30, 60,  29, 35,  15, 46,  10, 48,  27, 30,  56, 72,  72, 91,  41, 58,
       45, 50,  27, 49,  94, 110, 17, 39,  57, 99,  52, 81,  32, 56,  89, 92,
       0,  44,  59, 69,  93, 129, 34, 59,  32, 34,  39, 75,  0,  21,  10, 37,
       77, 113, 5,  28,  90, 111, 16, 55,  91, 109, 11, 32,  76, 92,  70, 76,
       41, 65,  5,  30,  57, 87,  21, 42,  39, 48,  96, 124, 51, 57,  36, 46,
       61, 83,  0,  25,  79, 87,  21, 28,  82, 131, 10, 32,  74, 116, 20, 64,
       61, 77,  26, 38,  98, 132, 36, 41,  8,  52,  64, 88,  67, 89,  87, 96,
       37, 60,  27, 68,  87, 97,  62, 97,  89, 105, 13, 47,  6,  11,  98, 135,
       85, 87,  17, 66,  12, 55,  87, 126, 26, 76,  19, 63,  91, 120, 9,  26,
       56, 58,  77, 83,  17, 60,  93, 126, 53, 54,  71, 72,  78, 81,  26, 35,
       87, 133, 25, 52,  92, 132, 74, 102, 80, 115, 66, 105, 37, 40,  7,  53,
       86, 102, 92, 135, 70, 107, 46, 48,  98, 103, 88, 117, 8,  53,  87, 88,
       52, 88,  19, 65,  56, 62,  86, 115, 37, 38,  91, 129, 13, 20,  78, 88,
       44, 74,  4,  26,  88, 89,  49, 75,  19, 55,  13, 45,  53, 67,  54, 62,
       21, 34,  60, 80,  58, 103, 35, 64,  62, 108, 22, 26,  65, 111, 94, 118,
       64, 100, 19, 69,  90, 100, 15, 23,  70, 79,  29, 79,  81, 104, 34, 40,
       23, 44,  39, 46,  58, 92,  88, 126, 29, 57,  53, 81,  89, 120, 41, 79,
       80, 98,  69, 113, 62, 89,  58, 71,  56, 96,  30, 33,  28, 47,  85, 113,
       67, 97,  96, 122, 71, 88,  43, 57,  79, 108, 15, 48,  4,  25,  15, 19,
       7,  12,  78, 121, 28, 37,  16, 36,  17, 29,  56, 100, 98, 141, 74, 114,
       9,  53,  99, 148, 63, 66,  36, 45,  55, 71,  85, 107, 60, 69,  36, 69,
       56, 86,  2,  48,  86, 108, 13, 34,  75, 94,  71, 82,  3,  31,  66, 85,
       86, 89,  76, 87,  3,  27,  5,  43,  23, 55,  70, 71,  91, 111, 98, 120,
       90, 104, 37, 84,  83, 106, 36, 56,  91, 92,  12, 19,  45, 64,  66, 80,
       34, 37,  37, 71,  45, 54,  48, 74,  95, 113, 96, 131, 98, 142, 20, 21,
       27, 71,  85, 129, 4,  50,  94, 106, 69, 77,  9,  13,  80, 97,  71, 120,
       7,  18,  40, 41,  10, 15,  8,  28,  25, 70,  88, 91,  87, 134, 66, 84,
       56, 74,  50, 63,  47, 90,  86, 106, 60, 63,  95, 99,  54, 86,  32, 74,
       79, 104, 87, 116, 55, 68,  33, 37,  37, 85,  92, 108, 94, 144, 7,  29,
       68, 84,  81, 124, 2,  45,  93, 137, 61, 62,  18, 27,  52, 89,  13, 55,
       12, 29,  54, 76,  11, 53,  97, 118, 54, 58,  8,  44,  52, 80,  62, 78,
       92, 98,  15, 63,  55, 67,  76, 88,  16, 59,  14, 34,  41, 82,  38, 86,
       48, 51,  48, 49,  61, 107, 34, 41,  77, 84,  41, 80,  57, 62,  79, 89,
       31, 56,  49, 91,  32, 36,  51, 60,  54, 75,  81, 98,  12, 30,  3,  16,
       56, 87,  19, 38,  78, 120, 60, 71,  95, 101, 85, 101, 56, 101, 46, 93,
       8,  57,  42, 63,  72, 97,  58, 105, 2,  5,   86, 127, 27, 60,  23, 45,
       12, 25,  54, 81,  73, 96,  77, 116, 80, 96,  70, 106, 62, 90,  67, 100,
       97, 126, 68, 78,  97, 123, 0,  36,  11, 20,  26, 57,  39, 79,  60, 106,
       44, 77,  10, 20,  17, 30,  82, 129, 25, 72,  15, 43,  35, 85,  30, 66,
       34, 53,  46, 73,  99, 136, 88, 111, 73, 99,  61, 76,  65, 112, 4,  42,
       15, 31,  42, 47,  40, 63,  39, 73,  2,  34,  34, 55,  37, 49,  36, 50,
       56, 92,  44, 85,  99, 104, 51, 95,  38, 74,  72, 104, 74, 84,  88, 98,
       68, 93,  59, 109, 62, 107, 22, 37,  34, 69,  72, 86,  24, 54,  6,  31,
       8,  41,  1,  33,  53, 71,  11, 22,  54, 73,  3,  20,  45, 51,  74, 91,
       43, 75,  63, 75,  36, 72,  67, 104, 84, 115, 36, 70,  99, 110, 63, 77,
       96, 130, 27, 74,  56, 103, 92, 122, 35, 84,  38, 88,  83, 117, 64, 85,
       58, 59,  22, 40,  96, 143, 95, 97,  59, 70,  84, 92,  58, 102, 28, 32,
       73, 117, 2,  42,  83, 125, 53, 96,  53, 62,  30, 62,  95, 132, 94, 143,
       80, 129, 14, 36,  31, 65,  44, 75,  53, 69,  23, 27,  20, 42,  66, 96,
       24, 60,  33, 42,  18, 57,  4,  28,  98, 146, 26, 43,  19, 61,  49, 76,
       60, 94,  65, 87,  19, 32,  34, 49,  49, 63,  46, 79,  3,  33,  93, 124,
       41, 77,  1,  31,  4,  41,  60, 81,  21, 57,  21, 60,  54, 98,  9,  54,
       42, 75,  47, 81,  82, 100, 17, 27,  26, 44,  82, 93,  50, 81,  19, 54,
       1,  46,  41, 68,  77, 119, 79, 123, 20, 34,  76, 115, 33, 79,  28, 73,
       4,  46,  96, 120, 8,  34,  35, 72,  18, 33,  81, 94,  45, 57,  6,  30,
       91, 141, 89, 123, 79, 84,  28, 71,  66, 90,  79, 106, 14, 19,  23, 34,
       31, 73,  90, 120, 4,  8,   24, 46,  42, 77,  37, 54,  49, 74,  90, 91,
       16, 45,  32, 61,  10, 42,  6,  48,  42, 60,  79, 110, 90, 133, 35, 40,
       41, 78,  31, 72,  48, 61,  69, 76,  55, 72,  11, 39,  34, 83,  72, 100,
       75, 78,  69, 110, 21, 67,  8,  36,  7,  54,  33, 83,  30, 37,  88, 103,
       77, 122, 68, 87,  70, 101, 29, 71,  26, 51,  0,  48,  81, 91,  4,  18,
       71, 86,  10, 11,  22, 57,  88, 125, 24, 73,  1,  29,  10, 29,  22, 72,
       97, 101, 93, 117, 28, 45,  87, 90,  41, 84,  62, 80,  89, 90,  13, 52,
       97, 139, 82, 107, 14, 54,  97, 124, 64, 71,  34, 63,  37, 78,  95, 100,
       37, 50,  75, 106, 77, 100, 92, 116, 42, 68,  50, 66,  96, 116, 15, 60,
       23, 72,  55, 59,  49, 62,  40, 51,  87, 113, 6,  41,  35, 53,  16, 64,
       0,  37,  66, 92,  72, 96,  38, 39,  75, 79,  9,  20,  58, 101, 99, 117,
       93, 110, 53, 80,  15, 61,  32, 52,  14, 63,  45, 94,  53, 56,  23, 54,
       27, 59,  51, 87,  35, 43,  48, 64,  42, 71,  86, 87,  58, 85,  97, 105,
       95, 138, 41, 63,  90, 110, 48, 58,  98, 143, 32, 44,  32, 35,  55, 56,
       34, 66,  53, 79,  40, 90,  4,  10,  27, 31,  20, 32,  81, 105, 81, 128,
       45, 81,  54, 101, 60, 68,  62, 75,  88, 135, 25, 58,  1,  4,   40, 56,
       28, 68,  34, 60,  27, 73,  75, 120, 81, 97,  84, 118, 2,  32,  45, 93,
       61, 102, 69, 99,  9,  59,  92, 104, 75, 113, 77, 101, 78, 125, 65, 94,
       86, 93,  48, 84,  52, 59,  42, 81,  6,  8,   96, 127, 11, 46,  22, 70,
       90, 139, 77, 118, 28, 51,  67, 76,  64, 72,  93, 131, 82, 128, 99, 134,
       72, 90,  28, 38,  82, 87,  66, 77,  88, 137, 13, 48,  93, 133, 53, 73,
       98, 131, 92, 121, 50, 77,  5,  39,  26, 64,  95, 140, 52, 70,  54, 102,
       33, 51,  61, 78,  31, 45,  64, 87,  48, 77,  24, 65,  76, 93,  96, 109,
       6,  56,  26, 27,  6,  53,  32, 49,  64, 77,  93, 108, 55, 57,  72, 103,
       7,  31,  43, 81,  50, 100, 51, 69,  13, 23,  6,  38,  11, 33,  59, 79,
       36, 71,  69, 98,  68, 109, 23, 56,  62, 88,  73, 90,  76, 82,  74, 108,
       60, 96,  34, 65,  44, 92,  40, 70,  11, 45,  11, 27,  38, 72,  79, 100,
       65, 102, 51, 84,  31, 41,  45, 74,  66, 115, 29, 48,  44, 66,  12, 23,
       76, 83,  39, 88,  10, 39,  34, 82,  13, 62,  4,  7,   24, 45,  58, 107,
       35, 41,  34, 58,  71, 91,  7,  17,  16, 56,  56, 89,  71, 106, 18, 46,
       48, 82,  14, 27,  89, 115, 37, 48,  73, 75,  54, 85,  38, 45,  52, 75,
       93, 142, 63, 68,  41, 67,  8,  27,  27, 29,  95, 133, 35, 51,  56, 75,
       56, 106, 54, 83,  73, 77,  53, 74,  23, 64,  71, 114, 66, 95,  94, 114,
       54, 95,  26, 37,  66, 109, 36, 60,  2,  33,  76, 89,  32, 65,  52, 71,
       80, 87,  25, 41,  8,  18,  30, 52,  23, 47,  40, 69,  22, 38,  71, 99,
       63, 84,  28, 29,  69, 114, 54, 74,  80, 109, 96, 136, 73, 87,  36, 76,
       82, 97,  8,  19,  4,  33,  64, 102, 17, 42,  81, 99,  21, 43,  81, 108,
       71, 81,  31, 51,  14, 49,  84, 105, 5,  25,  22, 67,  64, 67,  37, 55,
       34, 54,  86, 100, 8,  9,   80, 102, 77, 91,  29, 38,  52, 86,  82, 130,
       30, 43,  74, 124, 55, 75,  34, 68,  76, 124, 5,  36,  50, 90,  39, 78,
       3,  38,  79, 122, 11, 55,  40, 65,  63, 99,  93, 122, 84, 117, 11, 48,
       59, 97,  0,  19,  77, 98,  45, 47,  49, 57,  98, 109, 44, 90,  55, 58,
       43, 55,  40, 55,  98, 125, 86, 136, 47, 71,  78, 118, 40, 89,  76, 108,
       63, 65,  51, 88,  59, 99,  50, 71,  55, 64,  39, 49,  78, 107, 4,  22,
       92, 111, 19, 29,  55, 77,  66, 102, 84, 112, 20, 50,  87, 104, 10, 54,
       51, 101, 58, 60,  83, 129, 90, 136, 49, 54,  73, 102, 62, 93,  98, 117,
       41, 55,  50, 70,  96, 132, 29, 46,  99, 147, 65, 113, 20, 51,  25, 26,
       78, 91,  31, 47,  44, 62,  4,  35,  21, 30,  45, 65,  48, 89,  9,  30,
       84, 110, 5,  44,  9,  45,  92, 126, 70, 116, 53, 82,  16, 62,  9,  44});

  ASSERT_EQ(tree.size(), objects.size());

  llvm::DenseSet<Object> removedObjects;

  for (size_t removedIndex : removedIndices) {
    tree.remove(objects[removedIndex]);
    removedObjects.insert(objects[removedIndex]);
  }

  actualObjects.clear();
  actualObjects.append(tree.begin(), tree.end());

  llvm::SmallVector<Object> expectedObjects;

  for (const Object &object : objects) {
    if (!removedObjects.contains(object)) {
      expectedObjects.push_back(object);
    }
  }

  EXPECT_THAT(actualObjects,
              testing::UnorderedElementsAreArray(expectedObjects));
}
