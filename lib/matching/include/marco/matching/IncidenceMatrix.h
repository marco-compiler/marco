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

		bool operator==(const IncidenceMatrix& other) const;
		bool operator!=(const IncidenceMatrix& other) const;

		IncidenceMatrix& operator+=(const IncidenceMatrix& rhs);
    IncidenceMatrix operator+(const IncidenceMatrix& rhs);

    IncidenceMatrix& operator-=(const IncidenceMatrix& rhs);
    IncidenceMatrix operator-(const IncidenceMatrix& rhs);

		const MultidimensionalRange& getEquationRanges() const;
		const MultidimensionalRange& getVariableRanges() const;

    llvm::iterator_range<IndexesIterator> getIndexes() const;

		void apply(AccessFunction accessFunction);
		bool get(llvm::ArrayRef<long> indexes) const;
		void set(llvm::ArrayRef<long> indexes);
		void unset(llvm::ArrayRef<long> indexes);

		void clear();

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
