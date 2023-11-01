#ifndef MARCO_MODELING_ACCESSFUNCTION_H
#define MARCO_MODELING_ACCESSFUNCTION_H

#include "marco/Modeling/DimensionAccess.h"
#include "marco/Modeling/IndexSet.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/Support/Casting.h"
#include <variant>

namespace llvm
{
  class raw_ostream;
}

namespace marco::modeling
{
  /// The access function describes how an array variable is accessed.
  class AccessFunction
  {
    public:
      enum Kind
      {
        Empty,
        Constant,
        Generic,
        RotoTranslation,
        ZeroResults
      };

      static std::unique_ptr<AccessFunction> build(
          mlir::MLIRContext* context,
          unsigned int numOfDimensions,
          llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results);

      static std::unique_ptr<AccessFunction> build(mlir::AffineMap affineMap);

      static std::unique_ptr<AccessFunction> fromExtendedMap(
          mlir::AffineMap affineMap,
          const DimensionAccess::FakeDimensionsMap& fakeDimensionsMap);

      static llvm::SmallVector<std::unique_ptr<DimensionAccess>>
      convertAffineExpressions(llvm::ArrayRef<mlir::AffineExpr> expressions);

      AccessFunction(
          mlir::MLIRContext* context,
          unsigned int numOfDimensions,
          llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results,
          DimensionAccess::FakeDimensionsMap fakeDimensionsMap);

      explicit AccessFunction(mlir::AffineMap affineMap);

    protected:
      AccessFunction(
          Kind kind,
          mlir::MLIRContext* context,
          unsigned int numOfDimensions,
          llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results,
          DimensionAccess::FakeDimensionsMap fakeDimensionsMap =
              DimensionAccess::FakeDimensionsMap());

    public:
      AccessFunction(const AccessFunction& other);

      virtual ~AccessFunction();

      AccessFunction& operator=(const AccessFunction& other);

      AccessFunction& operator=(AccessFunction&& other);

      friend void swap(AccessFunction& first, AccessFunction& second);

      virtual std::unique_ptr<AccessFunction> clone() const;

      /// @name LLVM-style RTTI methods.
      /// {

      Kind getKind() const
      {
        return kind;
      }

      static bool classof(const AccessFunction* obj)
      {
        return obj->getKind() == Generic;
      }

      template<typename T>
      bool isa() const
      {
        return llvm::isa<T>(this);
      }

      template<typename T>
      T* cast()
      {
        return llvm::cast<T>(this);
      }

      template<typename T>
      const T* cast() const
      {
        return llvm::cast<T>(this);
      }

      template<typename T>
      T* dyn_cast()
      {
        return llvm::dyn_cast<T>(this);
      }

      template<typename T>
      const T* dyn_cast() const
      {
        return llvm::dyn_cast<T>(this);
      }

      /// }

      bool operator==(const AccessFunction& other) const;

      bool operator!=(const AccessFunction& other) const;

      llvm::raw_ostream& dump(llvm::raw_ostream& os) const;

      mlir::MLIRContext* getContext() const;

      /// Get the number of dimensions.
      virtual size_t getNumOfDims() const;

      /// Get the number of results.
      virtual size_t getNumOfResults() const;

      llvm::ArrayRef<std::unique_ptr<DimensionAccess>> getResults() const;

      bool isAffine() const;

      mlir::AffineMap getAffineMap() const;

      /// Check if the access function has an identity layout, that is if the
      /// i-th dimension accesses the i-th induction variable with offset 0
      /// (e.g. [i0][i1][i2]).
      virtual bool isIdentity() const;

      /// Combine the access function with another one.
      ///
      /// Examples:
      ///   [i1 + 3][i0 - 2] * [i1 + 6][i0 - 1] = [i0 + 4][i1 + 2]
      ///   [i1 + 3][i1 - 2] * [5][i0 - 1] = [5][i1 + 2]
      ///   [5][i0 + 3] * [i0 + 1][i1 - 1] = [6][i0 + 2]
      virtual std::unique_ptr<AccessFunction> combine(
          const AccessFunction& other) const;

      /// Check whether the access function is invertible.
      /// An access function is invertible if all the available induction
      /// variables are used.
      virtual bool isInvertible() const;

      /// Get the inverse access (if possible).
      /// Returns nullptr if the inverse function can't be computed.
      virtual std::unique_ptr<AccessFunction> inverse() const;

      /// Apply the access function to a point, in order to obtain the mapped
      /// indices.
      virtual IndexSet map(const Point& point) const;

      /// Apply the access function to an index set, in order to obtain the
      /// mapped indices.
      virtual IndexSet map(const IndexSet& indices) const;

      /// Apply the inverse of the access function to a set of indices.
      /// If the access function is not invertible, then the inverse indices
      /// are determined starting from a parent set.
      virtual IndexSet inverseMap(
          const IndexSet& accessedIndices,
          const IndexSet& parentIndices) const;

    protected:
      const DimensionAccess::FakeDimensionsMap& getFakeDimensionsMap() const;

      mlir::AffineMap getExtendedAffineMap(
          DimensionAccess::FakeDimensionsMap& fakeDimensionsMap) const;

      std::unique_ptr<AccessFunction> getWithAtLeastNumDimensions(
          unsigned int requestedDims) const;

    private:
      Kind kind;
      mlir::MLIRContext* context;
      unsigned int numOfDimensions;
      llvm::SmallVector<std::unique_ptr<DimensionAccess>> results;
      DimensionAccess::FakeDimensionsMap fakeDimensionsMap;
  };

  llvm::raw_ostream& operator<<(
      llvm::raw_ostream& os, const AccessFunction& obj);
}

#endif  // MARCO_MODELING_ACCESSFUNCTION_H
