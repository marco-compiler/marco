#ifndef MARCO_MODELING_POINT_H
#define MARCO_MODELING_POINT_H

#include <initializer_list>
#include <iostream>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

namespace marco::modeling::internal
{
  /// n-D point.
  class Point
  {
    public:
      using data_type = long;

    private:
      using Container = llvm::SmallVector<data_type, 3>;

    public:
      using const_iterator = Container::const_iterator;

      Point(data_type value);

      Point(std::initializer_list<data_type> values);

      Point(llvm::ArrayRef<data_type> values);

      bool operator==(const Point& other) const;

      bool operator!=(const Point& other) const;

      data_type operator[](size_t index) const;

      /// Get the number of dimensions.
      size_t rank() const;

      const_iterator begin() const;

      const_iterator end() const;

    private:
      Container values;
  };

  std::ostream& operator<<(std::ostream& stream, const Point& obj);
}

#endif // MARCO_MODELING_POINT_H
