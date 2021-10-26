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
		IncidenceMatrix(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

		const MultidimensionalRange& getEquationRanges() const;
		const MultidimensionalRange& getVariableRanges() const;

		void apply(AccessFunction accessFunction);
		bool get(llvm::ArrayRef<long> indexes) const;
		void set(llvm::ArrayRef<long> indexes);
		void unset(llvm::ArrayRef<long> indexes);

		void clear();

		IncidenceMatrix& operator+=(const IncidenceMatrix& rhs);

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
