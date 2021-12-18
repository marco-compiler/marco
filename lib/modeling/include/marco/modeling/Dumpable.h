#ifndef MARCO_MODELING_DUMPABLE_H
#define MARCO_MODELING_DUMPABLE_H

#include <iostream>

namespace marco::modeling::internal
{
  class Dumpable
  {
    public:
      virtual ~Dumpable();

      void dump() const;

      virtual void dump(std::ostream& os) const = 0;
  };
}

#endif // MARCO_MODELING_DUMPABLE_H
