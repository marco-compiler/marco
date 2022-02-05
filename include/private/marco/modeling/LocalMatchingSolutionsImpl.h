#ifndef MARCO_MODELING_LOCALMATCHINGSOLUTIONSIMPL_H
#define MARCO_MODELING_LOCALMATCHINGSOLUTIONSIMPL_H

#include "marco/modeling/LocalMatchingSolutions.h"
#include "marco/modeling/MCIM.h"

namespace marco::modeling::internal
{
  class LocalMatchingSolutions::ImplInterface
  {
    public:
      virtual ~ImplInterface();

      virtual MCIM& operator[](size_t index) = 0;

      virtual size_t size() const = 0;
  };
}

#endif // MARCO_MODELING_LOCALMATCHINGSOLUTIONSIMPL_H
