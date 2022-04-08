#ifndef MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_H
#define MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/StorageUniquer.h"
#include <map>

namespace mlir::modelica
{
  mlir::Attribute getAttr(mlir::Type type, llvm::APInt value);
  mlir::Attribute getAttr(mlir::Type type, llvm::APFloat value);

  mlir::Attribute getZeroAttr(mlir::Type type);

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
      using InverseFunction =  std::pair<llvm::StringRef, llvm::ArrayRef<unsigned int>>;
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
#include "marco/Dialect/Modelica/ModelicaAttributes.h.inc"

#endif // MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_H
