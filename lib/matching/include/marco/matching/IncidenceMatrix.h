#ifndef MARCO_MATCHING_INCIDENCEMATRIX_H
#define MARCO_MATCHING_INCIDENCEMATRIX_H

#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
#include <list>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/raw_os_ostream.h>

#include "AccessFunction.h"
#include "Range.h"

namespace marco::matching::detail
{
	class IncidenceMatrix
	{
		public:
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
              const MultidimensionalRange& equationRange,
              const MultidimensionalRange& variableRange,
              std::function<MultidimensionalRange::const_iterator(const MultidimensionalRange&)> initFunction);

      bool operator==(const IndexesIterator& it) const;
      bool operator!=(const IndexesIterator& it) const;
      IndexesIterator& operator++();
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

		IncidenceMatrix(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

    void dump() const;

    static IncidenceMatrix row(MultidimensionalRange variableRanges);
    static IncidenceMatrix column(MultidimensionalRange equationRanges);

		bool operator==(const IncidenceMatrix& other) const;
		bool operator!=(const IncidenceMatrix& other) const;

    IncidenceMatrix operator!() const;

		IncidenceMatrix& operator+=(const IncidenceMatrix& rhs);
    IncidenceMatrix operator+(const IncidenceMatrix& rhs) const;

    IncidenceMatrix& operator-=(const IncidenceMatrix& rhs);
    IncidenceMatrix operator-(const IncidenceMatrix& rhs) const;

		const MultidimensionalRange& getEquationRanges() const;
		const MultidimensionalRange& getVariableRanges() const;

    llvm::iterator_range<IndexesIterator> getIndexes() const;

		void apply(AccessFunction accessFunction);
		bool get(llvm::ArrayRef<long> indexes) const;
		void set(llvm::ArrayRef<long> indexes);
		void unset(llvm::ArrayRef<long> indexes);

    /**
     * Get the number of set elements.
     *
     * @return number of positions set to 1.
     */
    size_t size() const;

    /**
     * Remove all the set indexes.
     */
		void clear();

    /**
     * Check if all the elements are zero.
     *
     * @return true if all the elements are zero; false otherwise
     */
    bool isEmpty() const;

    /**
     * Flatten all the equation rows into a single one.
     *
     * @return matrix with 1 row and n columns, with n equal to the number of variables
     */
		IncidenceMatrix flattenEquations() const;

    /**
     * Flatten all the variable columns into a single one.
     *
     * @return matrix with n rows and 1 column, with n equal to the number of equations
     */
		IncidenceMatrix flattenVariables() const;

    /**
     * Keep only the equation rows that are specified by the filter.
     * In other words, set to zero all the rows that are also set to zero
     * in the vector filter.
     *
     * @param filter    equation filter
     * @return matrix with filtered equations
     */
    IncidenceMatrix filterEquations(const IncidenceMatrix& filter) const;

    /**
     * Keep only the variable columns that are specified by the filter.
     * In other words, set to zero all the columns that are also set to zero
     * in the vector filter.
     *
     * @param filter    equation filter
     * @return matrix with filtered equations
     */
    IncidenceMatrix filterVariables(const IncidenceMatrix& filter) const;

    bool isDisjoint(const IncidenceMatrix& other) const;

		private:
		std::pair<size_t, size_t> getMatrixIndexes(llvm::ArrayRef<long> indexes) const;

		MultidimensionalRange equationRanges;
		MultidimensionalRange variableRanges;
		boost::numeric::ublas::matrix<bool> data;
	};

	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const IncidenceMatrix& matrix);

	std::ostream& operator<<(
			std::ostream& stream, const IncidenceMatrix& matrix);
}

#endif	// MARCO_MATCHING_INCIDENCEMATRIX_H
