#include "marco/Modeling/RTree.h"
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
} // namespace

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
