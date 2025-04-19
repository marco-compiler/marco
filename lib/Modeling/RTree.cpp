#include "marco/Modeling/RTree.h"

using namespace ::marco::modeling::r_tree;

namespace marco::modeling::r_tree::impl {
MultidimensionalRange getMBR(const MultidimensionalRange &first,
                             const MultidimensionalRange &second) {
  assert(first.rank() == second.rank() &&
         "Can't compute the MBR between ranges on two different hyper-spaces");

  llvm::SmallVector<Range> ranges;
  ranges.reserve(first.rank());

  for (size_t i = 0, e = first.rank(); i < e; ++i) {
    ranges.emplace_back(std::min(first[i].getBegin(), second[i].getBegin()),
                        std::max(first[i].getEnd(), second[i].getEnd()));
  }

  return {std::move(ranges)};
}
} // namespace marco::modeling::r_tree::impl
