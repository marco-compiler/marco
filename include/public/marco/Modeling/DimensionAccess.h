#ifndef MARCO_MODELING_DIMENSIONACCESS_H
#define MARCO_MODELING_DIMENSIONACCESS_H

#include "marco/Modeling/IndexSet.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include <memory>

namespace llvm
{
  class raw_ostream;
}

namespace marco::modeling
{
  class DimensionAccess
  {
    public:
      enum class Kind
      {
        Constant,
        Dimension,
        Add,
        Sub,
        Mul,
        Div,
        Range,
        Indices
      };

      class Redirect
      {
        public:
          Redirect();

          explicit Redirect(std::unique_ptr<DimensionAccess> dimensionAccess);

          Redirect(const Redirect& other);

          Redirect(Redirect&& other);

          ~Redirect();

          Redirect& operator=(const Redirect& other);

          Redirect& operator=(Redirect&& other);

          friend void swap(Redirect& first, Redirect& second);

          bool operator==(const Redirect& other) const;

          bool operator!=(const Redirect& other) const;

          const DimensionAccess& operator*() const;

          const DimensionAccess* operator->() const;

        private:
          std::unique_ptr<DimensionAccess> dimensionAccess;
      };

      using FakeDimensionsMap = llvm::DenseMap<
          unsigned int, DimensionAccess::Redirect>;

      static std::unique_ptr<DimensionAccess> build(
          mlir::AffineExpr expression);

      static std::unique_ptr<DimensionAccess>
      getDimensionAccessFromExtendedMap(
          mlir::AffineExpr expression,
          const DimensionAccess::FakeDimensionsMap& fakeDimensionsMap);

    protected:
      DimensionAccess(Kind kind, mlir::MLIRContext* context);

    public:
      DimensionAccess(const DimensionAccess& other);

      virtual ~DimensionAccess();

      friend void swap(DimensionAccess& first, DimensionAccess& second);

      [[nodiscard]] virtual std::unique_ptr<DimensionAccess> clone() const = 0;

      /// @name LLVM-style RTTI methods.
      /// {

      [[nodiscard]] Kind getKind() const
      {
        return kind;
      }

      template<typename T>
      [[nodiscard]] bool isa() const
      {
        return llvm::isa<T>(this);
      }

      template<typename T>
      [[nodiscard]] T* cast()
      {
        return llvm::cast<T>(this);
      }

      template<typename T>
      [[nodiscard]] const T* cast() const
      {
        return llvm::cast<T>(this);
      }

      template<typename T>
      [[nodiscard]] T* dyn_cast()
      {
        return llvm::dyn_cast<T>(this);
      }

      template<typename T>
      [[nodiscard]] const T* dyn_cast() const
      {
        return llvm::dyn_cast<T>(this);
      }

      /// }

      [[nodiscard]] virtual bool operator==(
          const DimensionAccess& other) const = 0;

      [[nodiscard]] virtual bool operator!=(
          const DimensionAccess& other) const = 0;

      virtual llvm::raw_ostream& dump(
          llvm::raw_ostream& os,
          const llvm::DenseMap<IndexSet*, uint64_t>& indexSetsIds) const = 0;

      virtual void collectIndexSets(
          llvm::SmallVectorImpl<IndexSet*>& indexSets) const = 0;

      [[nodiscard]] virtual std::unique_ptr<DimensionAccess> operator+(
          const DimensionAccess& other) const;

      [[nodiscard]] virtual std::unique_ptr<DimensionAccess> operator-(
          const DimensionAccess& other) const;

      [[nodiscard]] virtual std::unique_ptr<DimensionAccess> operator*(
          const DimensionAccess& other) const;

      [[nodiscard]] virtual std::unique_ptr<DimensionAccess> operator/(
          const DimensionAccess& other) const;

      [[nodiscard]] mlir::MLIRContext* getContext() const;

      [[nodiscard]] virtual bool isAffine() const;

      [[nodiscard]] virtual mlir::AffineExpr getAffineExpr() const;

      [[nodiscard]] virtual mlir::AffineExpr getAffineExpr(
          unsigned int numOfDimensions,
          FakeDimensionsMap& fakeDimensionsMap) const = 0;

      [[nodiscard]] virtual IndexSet map(const Point& point) const = 0;

    private:
      Kind kind;
      mlir::MLIRContext* context;
  };
}

#endif // MARCO_MODELING_DIMENSIONACCESS_H
