#ifndef MARCO_UTILS_SHAPE_H
#define MARCO_UTILS_SHAPE_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <variant>

namespace marco
{

  class Shape
  {
    public:
    struct DimensionSize {
      template<typename T>
      using Container = llvm::SmallVector<T, 3>;
      using Undefined = std::monostate;
      using Ragged = std::unique_ptr<Container<DimensionSize>>;

      DimensionSize();
      DimensionSize(long int);
      DimensionSize(std::initializer_list<DimensionSize> ragged);
      DimensionSize(llvm::ArrayRef<DimensionSize> ragged);

      DimensionSize(const DimensionSize& other);
      DimensionSize(DimensionSize&& other);

      static DimensionSize makeUndefined()
      {
        return DimensionSize();
      }

      ~DimensionSize();

      DimensionSize& operator=(const DimensionSize& other);
      DimensionSize& operator=(DimensionSize&& other);

      friend void swap(DimensionSize& first, DimensionSize& second);

      bool operator==(const DimensionSize& other) const;
      bool operator!=(const DimensionSize& other) const;

      bool isRagged() const;
      bool isUndefined() const;
      bool isParameter() const;

      long getNumericValue() const;

      friend llvm::hash_code hash_value(const DimensionSize& d);

		  friend std::string toString(const DimensionSize& size);
			friend std::ostream& operator<<(std::ostream& stream, const DimensionSize &size);

      llvm::ArrayRef<DimensionSize> asRagged() const;
		
		private:
      std::variant<Undefined, long, Ragged> value;
    };

    using Sizes = llvm::SmallVector<DimensionSize, 3>;
    using iterator = Sizes::iterator;
    using const_iterator = Sizes::const_iterator;

    iterator begin();
    iterator end();

    const_iterator begin() const;
    const_iterator end() const;

    DimensionSize& operator[](int index);
    const DimensionSize& operator[](int index) const;

    bool empty() const;
    size_t size() const;

    void push_back(const DimensionSize& d);

    bool operator==(const Shape& other) const;
    bool operator!=(const Shape& other) const;

    llvm::SmallVector<long, 3> to_old() const;

    bool isRagged() const;
    bool isUndefined() const;
    bool isParameter() const;

    Shape() = default;

    Shape(std::initializer_list<DimensionSize> values);
    Shape(llvm::ArrayRef<long> values);
    Shape(const llvm::SmallVector<long, 3>& values);

    Shape(llvm::NoneType);
    Shape(DimensionSize el);
    Shape(DimensionSize el, size_t length);

    llvm::ArrayRef<DimensionSize> dimensions() const;

		friend llvm::hash_code hash_value(const Shape& s);

    private:
    Sizes sizes;
  };

  extern std::string toString(const Shape& shape);

  extern std::vector<std::vector<long>> generateAllIndexes(Shape shape);
}// namespace marco

#endif//MARCO_UTILS_SHAPE_H