#ifndef MARCO_MATCHING_MCIM_H
#define MARCO_MATCHING_MCIM_H

#include <llvm/ADT/iterator_range.h>

#include "AccessFunction.h"
#include "MCIS.h"
#include "Range.h"

namespace marco::matching::detail
{
	/**
	 * Multidimensional Compressed Index Map (MCIM).
	 *
	 * It replaces the multidimensional incidence matrices in order to achieve
	 * O(1) scaling.
	 */
	class MCIM
	{
		public:
    class Impl;

    class IndexesIterator
    {
      public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = llvm::ArrayRef<long>;
      using difference_type = std::ptrdiff_t;
      using pointer = llvm::ArrayRef<long>*;
      using reference = llvm::ArrayRef<long>&;

      using Iterator = MultidimensionalRange::const_iterator;

      IndexesIterator(
              const MultidimensionalRange& equationRanges,
              const MultidimensionalRange& variableRanges,
              std::function<MultidimensionalRange::const_iterator(const MultidimensionalRange&)> initFunction);

      bool operator==(const IndexesIterator &it) const;
      bool operator!=(const IndexesIterator &it) const;
      IndexesIterator &operator++();
      IndexesIterator operator++(int);
      llvm::ArrayRef<long> operator*() const;

      private:
      void advance();

      size_t eqRank;
      Iterator eqCurrentIt;
      Iterator eqEndIt;
      Iterator varBeginIt;
      Iterator varCurrentIt;
      Iterator varEndIt;
      llvm::SmallVector<long, 4> indexes;
    };

		MCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

    MCIM(const MCIM& other);

    ~MCIM();

    const MultidimensionalRange& getEquationRanges() const;
    const MultidimensionalRange& getVariableRanges() const;

    llvm::iterator_range<IndexesIterator> getIndexes() const;

    void apply(const AccessFunction& access);
    bool get(llvm::ArrayRef<long> indexes) const;
    void set(llvm::ArrayRef<long> indexes);

    bool empty() const;
    void clear();

    MCIS flattenEquations() const;
    MCIS flattenVariables() const;

    MCIM filterEquations(const MCIS& filter) const;
    MCIM filterVariables(const MCIS& filter) const;

    std::vector<MCIM> splitGroups() const;

		private:
    MCIM(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl;
	};

  std::ostream& operator<<(std::ostream& stream, const MCIM& mcim);
}

#endif	// MARCO_MATCHING_MCIM_H
