#ifndef MARCO_MODELING_MCIM_H
#define MARCO_MODELING_MCIM_H

#include "marco/Modeling/MCIMGroup.h"
#include <memory>

namespace llvm {
class raw_ostream;
}

namespace marco::modeling::internal {
/// Multidimensional Compressed Index Map (MCIM).
/// It replaces the multidimensional incidence matrices in order to achieve O(1)
/// scaling.
class MCIM {
public:
  class IndicesIterator {
  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = std::pair<Point, Point>;
    using difference_type = std::ptrdiff_t;
    using pointer = std::pair<Point, Point> *;
    using reference = std::pair<Point, Point> &;

    using Iterator = IndexSet::const_point_iterator;

    IndicesIterator(
        const IndexSet &equationRanges, const IndexSet &variableRanges,
        llvm::function_ref<IndexSet::const_point_iterator(const IndexSet &)>
            initFunction);

    bool operator==(const IndicesIterator &it) const;

    bool operator!=(const IndicesIterator &it) const;

    IndicesIterator &operator++();

    IndicesIterator operator++(int);

    value_type operator*() const;

  private:
    void advance();

    Iterator eqCurrentIt;
    Iterator eqEndIt;
    Iterator varBeginIt;
    Iterator varCurrentIt;
    Iterator varEndIt;
  };

  MCIM(MultidimensionalRange equationRanges,
       MultidimensionalRange variableRanges);

  MCIM(IndexSet equationRanges, IndexSet variableRanges);

  MCIM(std::shared_ptr<const IndexSet> equationRanges,
       std::shared_ptr<const IndexSet> variableRanges);

public:
  MCIM(const MCIM &other);

  MCIM(MCIM &&other);

  ~MCIM();

  MCIM &operator=(const MCIM &other);

  MCIM &operator=(MCIM &&other);

  friend void swap(MCIM &first, MCIM &second);

  bool operator==(const MCIM &other) const;

  bool operator!=(const MCIM &other) const;

  const IndexSet &getEquationSpace() const;

  const IndexSet &getVariableSpace() const;

  IndicesIterator indicesBegin() const;

  IndicesIterator indicesEnd() const;

  MCIM &operator+=(const MCIM &rhs);

  MCIM operator+(const MCIM &rhs) const;

  MCIM &operator-=(const MCIM &rhs);

  MCIM operator-(const MCIM &rhs) const;

  void apply(const AccessFunction &access);

  void apply(const MultidimensionalRange &equations,
             const AccessFunction &access);

  void apply(const IndexSet &equations, const AccessFunction &access);

  bool get(const Point &equation, const Point &variable) const;

  void set(const Point &equation, const Point &variable);

  void unset(const Point &equation, const Point &variable);

  bool empty() const;

  void clear();

  IndexSet flattenRows() const;

  IndexSet flattenColumns() const;

  MCIM filterRows(const IndexSet &filter) const;

  MCIM filterColumns(const IndexSet &filter) const;

  std::vector<MCIM> splitGroups() const;

private:
  MCIMGroup &addGroup(std::unique_ptr<MCIMGroup> group);

private:
  std::shared_ptr<const IndexSet> equationSpace;
  std::shared_ptr<const IndexSet> variableSpace;
  std::vector<std::unique_ptr<MCIMGroup>> groups;
  IndexSet points;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const MCIM &obj);
} // namespace marco::modeling::internal

#endif // MARCO_MODELING_MCIM_H
