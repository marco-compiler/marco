#ifndef MARCO_MATCHING_DUMPABLE_H
#define MARCO_MATCHING_DUMPABLE_H

#include <iostream>

namespace marco::matching::detail
{
  class Dumpable
  {
    public:
    void dump() const;

    virtual void dump(std::ostream& os) const = 0;
  };
}

#endif //MARCO_MATCHING_DUMPABLE_H
