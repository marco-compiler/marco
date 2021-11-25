#include <marco/matching/Dumpable.h>

using namespace marco::matching::detail;

void Dumpable::dump() const
{
  dump(std::clog);
}
