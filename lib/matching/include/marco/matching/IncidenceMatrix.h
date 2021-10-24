#ifndef MARCO_MATCHING_INCIDENCEMATRIX_H
#define MARCO_MATCHING_INCIDENCEMATRIX_H

#include <boost/numeric/ublas/matrix.hpp>

#include "AccessFunction.h"
#include "MCIS.h"
#include "Range.h"

namespace marco::matching
{
	class IncidenceMatrix
	{
		public:
		IncidenceMatrix(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges);

		void apply(AccessFunction accessFunction);
		bool get(llvm::ArrayRef<long> indexes) const;
		void set(llvm::ArrayRef<long> indexes);
		void unset(llvm::ArrayRef<long> indexes);

		private:
		void splitIndexes(
				llvm::ArrayRef<long> indexes,
				llvm::SmallVectorImpl<size_t>& equationIndexes,
				llvm::SmallVectorImpl<size_t>& variableIndexes) const;

		std::pair<size_t, size_t> getMatrixIndexes(llvm::ArrayRef<long> indexes) const;

		MultidimensionalRange equationRanges;
		MultidimensionalRange variableRanges;
		boost::numeric::ublas::matrix<bool> data;
	};
}

#endif	// MARCO_MATCHING_INCIDENCEMATRIX_H
