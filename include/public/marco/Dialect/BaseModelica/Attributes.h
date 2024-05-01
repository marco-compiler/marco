#ifndef MARCO_DIALECTS_BASEMODELICA_ATTRIBUTES_H
#define MARCO_DIALECTS_BASEMODELICA_ATTRIBUTES_H

#include "marco/Dialect/BaseModelica/EquationPath.h"
#include "marco/Dialect/BaseModelica/ExpressionPath.h"
#include "marco/Dialect/BaseModelica/AttrInterfaces.h"
#include "marco/Dialect/BaseModelica/Types.h"
#include "marco/Dialect/Modeling/Attributes.h"
#include "marco/Modeling/Scheduling.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include <map>

namespace mlir::bmodelica
{
  using RangeAttr = ::mlir::modeling::RangeAttr;
  using MultidimensionalRangeAttr = ::mlir::modeling::MultidimensionalRangeAttr;
  using IndexSetAttr = ::mlir::modeling::IndexSetAttr;

  using EquationScheduleDirection = ::marco::modeling::scheduling::Direction;

  mlir::Attribute getAttr(mlir::Type type, int64_t value);
  mlir::Attribute getAttr(mlir::Type type, double value);

  mlir::Attribute getAttr(ArrayType arrayType, llvm::ArrayRef<int64_t> values);
  mlir::Attribute getAttr(ArrayType arrayType, llvm::ArrayRef<double> values);

  namespace detail
  {
    template<typename ValueType, typename BaseIterator>
    class InvertibleArgumentsIterator
    {
      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = ValueType;
        using difference_type = std::ptrdiff_t;
        using pointer = ValueType*;
        using reference = ValueType&;

        InvertibleArgumentsIterator(BaseIterator iterator) : iterator(iterator)
        {
        }

        operator bool() const { return iterator(); }

        bool operator==(const InvertibleArgumentsIterator& it) const
        {
          return it.iterator == iterator;
        }

        bool operator!=(const InvertibleArgumentsIterator& it) const
        {
          return it.iterator != iterator;
        }

        InvertibleArgumentsIterator& operator++()
        {
          iterator++;
          return *this;
        }

        InvertibleArgumentsIterator operator++(int)
        {
          auto temp = *this;
          iterator++;
          return temp;
        }

        value_type operator*() const
        {
          return iterator->first;
        }

      private:
        BaseIterator iterator;
    };
  }

  class InverseFunctionsMap
  {
    private:
    llvm::StringRef str;
      using InverseFunction = std::pair<llvm::StringRef, llvm::ArrayRef<unsigned int>>;
      using Map = std::map<unsigned int, InverseFunction>;

    public:
      using iterator = detail::InvertibleArgumentsIterator<unsigned int, Map::iterator>;
      using const_iterator = detail::InvertibleArgumentsIterator<unsigned int, Map::const_iterator>;

      bool operator==(const InverseFunctionsMap& other) const;

      InverseFunction& operator[](unsigned int arg);

      bool empty() const;

      iterator begin();
      const_iterator begin() const;

      iterator end();
      const_iterator end() const;

      InverseFunctionsMap allocateInto(mlir::StorageUniquer::StorageAllocator& allocator);

      bool isInvertible(unsigned int argumentIndex) const;
      llvm::StringRef getFunction(unsigned int argumentIndex) const;
      llvm::ArrayRef<unsigned int> getArgumentsIndexes(unsigned int argumentIndex) const;

    private:
      Map map;
  };

  llvm::hash_code hash_value(const InverseFunctionsMap& map);
}

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/BaseModelica/BaseModelicaAttributes.h.inc"

#endif // MARCO_DIALECTS_BASEMODELICA_ATTRIBUTES_H
