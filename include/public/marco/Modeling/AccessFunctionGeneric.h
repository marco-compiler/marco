#ifndef MARCO_MODELING_ACCESSFUNCTIONGENERIC_H
#define MARCO_MODELING_ACCESSFUNCTIONGENERIC_H

#include "marco/Modeling/AccessFunction.h"

namespace marco::modeling
{
  class AccessFunctionGeneric : public AccessFunction
  {
    public:
      AccessFunctionGeneric(
          mlir::MLIRContext* context,
          uint64_t numOfDimensions,
          llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results,
          DimensionAccess::FakeDimensionsMap fakeDimensionsMap);

      explicit AccessFunctionGeneric(mlir::AffineMap affineMap);

    protected:
      AccessFunctionGeneric(
          Kind kind,
          mlir::MLIRContext* context,
          uint64_t numOfDimensions,
          llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results,
          DimensionAccess::FakeDimensionsMap fakeDimensionsMap);

    public:
      AccessFunctionGeneric(const AccessFunctionGeneric& other);

      AccessFunctionGeneric(AccessFunctionGeneric&& other);

      ~AccessFunctionGeneric() override;

      AccessFunctionGeneric& operator=(const AccessFunctionGeneric& other);

      AccessFunctionGeneric& operator=(AccessFunctionGeneric&& other);

      friend void swap(
          AccessFunctionGeneric& first, AccessFunctionGeneric& second);

      [[nodiscard]] std::unique_ptr<AccessFunction> clone() const override;

      /// @name LLVM-style RTTI methods
      /// {

      static bool classof(const AccessFunction* obj)
      {
        return obj->getKind() == Generic;
      }

      /// }

      bool operator==(const AccessFunction& other) const override;

      bool operator==(const AccessFunctionGeneric& other) const;

      bool operator!=(const AccessFunction& other) const override;

      bool operator!=(const AccessFunctionGeneric& other) const;

      llvm::raw_ostream& dump(llvm::raw_ostream& os) const override;

      [[nodiscard]] uint64_t getNumOfDims() const override;

      [[nodiscard]] uint64_t getNumOfResults() const override;

      [[nodiscard]] bool isAffine() const override;

      [[nodiscard]] mlir::AffineMap getAffineMap() const override;

      [[nodiscard]] bool isIdentity() const override;

      [[nodiscard]] IndexSet map(const Point& point) const override;

      [[nodiscard]] IndexSet map(const IndexSet& indices) const override;

      [[nodiscard]] llvm::SmallVector<std::unique_ptr<DimensionAccess>, 6>
      getGeneralizedAccesses() const override;

      [[nodiscard]] mlir::AffineMap getExtendedAffineMap(
          DimensionAccess::FakeDimensionsMap& fakeDimensionsMap) const override;

      [[nodiscard]] llvm::ArrayRef<std::unique_ptr<DimensionAccess>>
      getResults() const;

    private:
      uint64_t numOfDimensions;
      llvm::SmallVector<std::unique_ptr<DimensionAccess>> results;
      DimensionAccess::FakeDimensionsMap fakeDimensionsMap;
  };
}

#endif // MARCO_MODELING_ACCESSFUNCTIONGENERIC_H