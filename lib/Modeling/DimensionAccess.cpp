#include "marco/Modeling/DimensionAccess.h"
#include "marco/Modeling/DimensionAccessAdd.h"
#include "marco/Modeling/DimensionAccessConstant.h"
#include "marco/Modeling/DimensionAccessDimension.h"
#include "marco/Modeling/DimensionAccessDiv.h"
#include "marco/Modeling/DimensionAccessMul.h"
#include "marco/Modeling/DimensionAccessSub.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  DimensionAccess::Redirect::Redirect()
  {
  }

  DimensionAccess::Redirect::Redirect(
      std::unique_ptr<DimensionAccess> dimensionAccess)
      : dimensionAccess(std::move(dimensionAccess))
  {
  }

  DimensionAccess::Redirect::Redirect(const Redirect& other)
      : dimensionAccess(other.dimensionAccess->clone())
  {
  }

  DimensionAccess::Redirect::Redirect(
      DimensionAccess::Redirect&& other) = default;

  DimensionAccess::Redirect::~Redirect() = default;

  DimensionAccess::Redirect& DimensionAccess::Redirect::operator=(
      const DimensionAccess::Redirect& other)
  {
    DimensionAccess::Redirect result(other);
    swap(*this, result);
    return *this;
  }

  DimensionAccess::Redirect& DimensionAccess::Redirect::operator=(
      DimensionAccess::Redirect&& other) = default;

  void swap(
      DimensionAccess::Redirect& first,
      DimensionAccess::Redirect& second)
  {
    using std::swap;
    swap(first.dimensionAccess, second.dimensionAccess);
  }

  bool DimensionAccess::Redirect::operator==(const Redirect& other) const
  {
    return *dimensionAccess == *other.dimensionAccess;
  }

  bool DimensionAccess::Redirect::operator!=(const Redirect& other) const
  {
    return !(*this == other);
  }

  const DimensionAccess& DimensionAccess::Redirect::operator*() const
  {
    assert(dimensionAccess && "Dimension access not set");
    return *dimensionAccess;
  }

  const DimensionAccess* DimensionAccess::Redirect::operator->() const
  {
    assert(dimensionAccess && "Dimension access not set");
    return dimensionAccess.get();
  }

  std::unique_ptr<DimensionAccess> DimensionAccess::build(
      mlir::AffineExpr expression)
  {
    if (auto constantExpr = expression.dyn_cast<mlir::AffineConstantExpr>()) {
      return std::make_unique<DimensionAccessConstant>(
          constantExpr.getContext(), constantExpr.getValue());
    }

    if (auto dimExpr = expression.dyn_cast<mlir::AffineDimExpr>()) {
      return std::make_unique<DimensionAccessDimension>(
          dimExpr.getContext(), dimExpr.getPosition());
    }

    if (auto binaryExpr = expression.dyn_cast<mlir::AffineBinaryOpExpr>()) {
      auto kind = binaryExpr.getKind();

      if (kind == mlir::AffineExprKind::Add) {
        return std::make_unique<DimensionAccessAdd>(
            binaryExpr.getContext(),
            DimensionAccess::build(binaryExpr.getLHS()),
            DimensionAccess::build(binaryExpr.getRHS()));
      }

      if (kind == mlir::AffineExprKind::Mul) {
        return std::make_unique<DimensionAccessMul>(
            binaryExpr.getContext(),
            DimensionAccess::build(binaryExpr.getLHS()),
            DimensionAccess::build(binaryExpr.getRHS()));
      }

      if (kind == mlir::AffineExprKind::FloorDiv) {
        return std::make_unique<DimensionAccessDiv>(
            binaryExpr.getContext(),
            DimensionAccess::build(binaryExpr.getLHS()),
            DimensionAccess::build(binaryExpr.getRHS()));
      }
    }

    llvm_unreachable("Unexpected expression type");
    return nullptr;
  }

  std::unique_ptr<DimensionAccess>
  DimensionAccess::getDimensionAccessFromExtendedMap(
      mlir::AffineExpr expression,
      const DimensionAccess::FakeDimensionsMap& fakeDimensionsMap)
  {
    if (auto constantExpr = expression.dyn_cast<mlir::AffineConstantExpr>()) {
      return std::make_unique<DimensionAccessConstant>(
          constantExpr.getContext(), constantExpr.getValue());
    }

    if (auto dimExpr = expression.dyn_cast<mlir::AffineDimExpr>()) {
      auto fakeDimReplacementIt = fakeDimensionsMap.find(dimExpr.getPosition());

      if (fakeDimReplacementIt != fakeDimensionsMap.end()) {
        return fakeDimReplacementIt->getSecond()->clone();
      }

      return std::make_unique<DimensionAccessDimension>(
          dimExpr.getContext(), dimExpr.getPosition());
    }

    if (auto binaryExpr = expression.dyn_cast<mlir::AffineBinaryOpExpr>()) {
      auto kind = binaryExpr.getKind();

      if (kind == mlir::AffineExprKind::Add) {
        return std::make_unique<DimensionAccessAdd>(
            binaryExpr.getContext(),
            DimensionAccess::getDimensionAccessFromExtendedMap(
                binaryExpr.getLHS(), fakeDimensionsMap),
            DimensionAccess::getDimensionAccessFromExtendedMap(
                binaryExpr.getRHS(), fakeDimensionsMap));
      }

      if (kind == mlir::AffineExprKind::Mul) {
        return std::make_unique<DimensionAccessMul>(
            binaryExpr.getContext(),
            DimensionAccess::getDimensionAccessFromExtendedMap(
                binaryExpr.getLHS(), fakeDimensionsMap),
            DimensionAccess::getDimensionAccessFromExtendedMap(
                binaryExpr.getRHS(), fakeDimensionsMap));
      }

      if (kind == mlir::AffineExprKind::FloorDiv) {
        return std::make_unique<DimensionAccessDiv>(
            binaryExpr.getContext(),
            DimensionAccess::getDimensionAccessFromExtendedMap(
                binaryExpr.getLHS(), fakeDimensionsMap),
            DimensionAccess::getDimensionAccessFromExtendedMap(
                binaryExpr.getRHS(), fakeDimensionsMap));
      }
    }

    llvm_unreachable("Unexpected expression type");
    return nullptr;
  }

  DimensionAccess::DimensionAccess(Kind kind, mlir::MLIRContext* context)
      : kind(kind),
        context(context)
  {
  }

  DimensionAccess::DimensionAccess(const DimensionAccess& other) = default;

  DimensionAccess::~DimensionAccess() = default;

  void swap(DimensionAccess& first, DimensionAccess& second)
  {
    using std::swap;
    swap(first.kind, second.kind);
    swap(first.context, second.context);
  }

  std::unique_ptr<DimensionAccess> DimensionAccess::operator+(
      const DimensionAccess& other) const
  {
    return std::make_unique<DimensionAccessAdd>(
        getContext(), this->clone(), other.clone());
  }

  std::unique_ptr<DimensionAccess> DimensionAccess::operator-(
      const DimensionAccess& other) const
  {
    return std::make_unique<DimensionAccessSub>(
        getContext(), this->clone(), other.clone());
  }

  std::unique_ptr<DimensionAccess> DimensionAccess::operator*(
      const DimensionAccess& other) const
  {
    return std::make_unique<DimensionAccessMul>(
        getContext(), this->clone(), other.clone());
  }

  std::unique_ptr<DimensionAccess> DimensionAccess::operator/(
      const DimensionAccess& other) const
  {
    return std::make_unique<DimensionAccessDiv>(
        getContext(), this->clone(), other.clone());
  }

  mlir::MLIRContext* DimensionAccess::getContext() const
  {
    assert(context && "MLIR context not set");
    return context;
  }

  bool DimensionAccess::isAffine() const
  {
    return false;
  }

  mlir::AffineExpr DimensionAccess::getAffineExpr() const
  {
    llvm_unreachable("Not an affine expression");
    return nullptr;
  }

  llvm::raw_ostream& operator<<(
      llvm::raw_ostream& os, const DimensionAccess& dimensionAccess)
  {
    return dimensionAccess.dump(os);
  }
}
