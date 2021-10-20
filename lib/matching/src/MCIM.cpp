#include <marco/matching/MCIM.h>

using namespace marco::matching;

MCIM::MCIM(MultidimensionalRange equationRanges, MultidimensionalRange variableRanges)
		: equationRanges(equationRanges), variableRanges(variableRanges)
{
	size_t size = 1;
	size *= equationRanges.flatSize();
	size *= variableRanges.flatSize();
	data = new bool[size];
}

MCIM::~MCIM()
{
	delete[] data;
}
