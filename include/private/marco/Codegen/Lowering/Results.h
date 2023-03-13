#ifndef MARCO_CODEGEN_LOWERING_RESULTS_H
#define MARCO_CODEGEN_LOWERING_RESULTS_H

#include "marco/Codegen/Lowering/Reference.h"
#include <vector>

namespace marco::codegen::lowering
{
  class Result
  {
    public:
      explicit Result(Reference reference);

      /// @name Forwarding methods.
      /// {

      mlir::Location getLoc() const;

      mlir::Value getReference() const;

      mlir::Value get(mlir::Location loc) const;

      void set(mlir::Location loc, mlir::Value value);

      /// }

      mlir::Value operator*() const;

    private:
      Reference reference;
  };

  class Results
  {
    private:
      using Container = std::vector<Result>;

    public:
      using iterator = typename Container::iterator;
      using const_iterator = typename Container::const_iterator;

      /// Default constructor for empty results.
      Results();

      /// Constructor for just one result.
      Results(Reference value);

      /// Constructor for multiple results.
      template<typename It>
      Results(It beginIt, It endIt) : values(beginIt, endIt)
      {
      }

      Result& operator[](size_t index);
      const Result& operator[](size_t index) const;

      /// Append a new result to the list of current ones.
      void append(Reference value);

      /// Append a new result to the list of current ones.
      void append(Result value);

      /// Get the number of results.
      size_t size() const;

      /// @name Iterators
      /// {

      iterator begin();
      const_iterator begin() const;

      iterator end();
      const_iterator end() const;

      /// }

    private:
      Container values;
  };
}

#endif // MARCO_CODEGEN_LOWERING_RESULTS_H
