#ifndef MARCO_CODEGEN_LOWERING_RESULTS_H
#define MARCO_CODEGEN_LOWERING_RESULTS_H

#include "marco/Codegen/Lowering/Reference.h"
#include <vector>

namespace marco::codegen::lowering
{
  class Results
  {
    private:
      using Container = std::vector<Reference>;

    public:
      using iterator = typename Container::iterator;
      using const_iterator = typename Container::const_iterator;

      Results();
      Results(Reference value);

      template<typename It>
      Results(It beginIt, It endIt) : values(beginIt, endIt)
      {
      }

      Reference& operator[](size_t index);
      const Reference& operator[](size_t index) const;

      void append(Reference value);

      size_t size() const;

      iterator begin();
      const_iterator begin() const;

      iterator end();
      const_iterator end() const;

    private:
      Container values;
  };
}

#endif // MARCO_CODEGEN_LOWERING_RESULTS_H
