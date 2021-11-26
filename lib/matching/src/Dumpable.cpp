#include <marco/matching/Dumpable.h>

using namespace marco::matching::detail;

Dumpable::~Dumpable() = default;

void Dumpable::dump() const
{
  dump(std::clog);
}
