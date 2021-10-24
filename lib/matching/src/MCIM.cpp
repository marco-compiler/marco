#include <marco/matching/MCIM.h>

using namespace marco::matching;

MCIMElement::MCIMElement(long delta, MCIS k)
		: delta(delta), k(std::move(k))
{
}

MCIM::MCIM()
{
}
