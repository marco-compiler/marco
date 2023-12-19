#ifndef MARCO_DIALECTS_SBG_ATTRIBUTES_H
#define MARCO_DIALECTS_SBG_ATTRIBUTES_H

#include "sbg/pw_map.hpp"
#include "marco/Dialect/Modelica/EquationPath.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/Hashing.h"

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/SBG/SBGAttributes.h.inc"

namespace mlir::sbg {
  using MDNat = ::SBG::Util::MD_NAT;
  using Rational = ::SBG::Util::RATIONAL;
  using Interval = ::SBG::LIB::Interval;
  using MDI = ::SBG::LIB::MultiDimInter;
  using OrdSet = ::SBG::LIB::OrdSet;
  using Set = ::SBG::LIB::UnordSet;
  using LinearExp = ::SBG::LIB::LExp;
  using Exp = ::SBG::LIB::Exp;
  using OrdDomMap = ::SBG::LIB::CanonMap;
  using Map = ::SBG::LIB::BaseMap;
  using OrdDomPWMap = ::SBG::LIB::CanonPWMap;
  using PWMap = ::SBG::LIB::BasePWMap;
  using EqPath = ::mlir::modelica::EquationPath;
}

namespace mlir
{
  template<>
  struct FieldParser<sbg::MDNat>
  {
    static FailureOr<sbg::MDNat>
    parse(AsmParser& parser);
  };

  AsmPrinter& operator<<(
    AsmPrinter&printer, const sbg::MDNat& n);

  template<>
  struct FieldParser<sbg::Rational>
  {
    static FailureOr<sbg::Rational>
    parse(AsmParser& parser);
  };

  AsmPrinter& operator<<(
    AsmPrinter& printer, const sbg::Rational& r);

  template<>
  struct FieldParser<std::optional<sbg::Rational>>
  {
    static FailureOr<std::optional<sbg::Rational>>
    parse(AsmParser& parser);
  };

  AsmPrinter& operator<<(
      AsmPrinter& printer, const std::optional<sbg::Rational>& r);

  template<>
  struct FieldParser<sbg::Interval>
  {
    static FailureOr<sbg::Interval>
    parse(AsmParser& parser);
  };

  AsmPrinter& operator<<(
    AsmPrinter& printer, const sbg::Interval& i);

  template<>
  struct FieldParser<sbg::MDI>
  {
    static FailureOr<sbg::MDI>
    parse(AsmParser& parser);
  };

  AsmPrinter& operator<<(
    AsmPrinter& printer, const sbg::MDI& mdi);

  template<>
  struct FieldParser<sbg::OrdSet>
  {
    static FailureOr<sbg::OrdSet>
    parse(AsmParser& parser);
  };

  AsmPrinter& operator<<(
    AsmPrinter& printer, const sbg::OrdSet& ord_set);

  template<>
  struct FieldParser<sbg::Set>
  {
    static FailureOr<sbg::Set>
    parse(AsmParser& parser);
  };

  AsmPrinter& operator<<(
    AsmPrinter& printer, const sbg::Set& unord_set);

  template<>
  struct FieldParser<sbg::LinearExp>
  {
    static FailureOr<sbg::LinearExp>
    parse(AsmParser& parser);
  };

  AsmPrinter& operator<<(
    AsmPrinter& printer, const sbg::LinearExp& le);

  template<>
  struct FieldParser<sbg::Exp>
  {
    static FailureOr<sbg::Exp>
    parse(AsmParser& parser);
  };

  AsmPrinter& operator<<(
    AsmPrinter &printer, const sbg::Exp &e);

  template<>
  struct FieldParser<sbg::OrdDomMap>
  {
    static FailureOr<sbg::OrdDomMap>
    parse(AsmParser& parser);
  };

  AsmPrinter& operator<<(
    AsmPrinter& printer, const sbg::OrdDomMap& m);

  template<>
  struct FieldParser<sbg::Map>
  {
    static FailureOr<sbg::Map>
    parse(AsmParser& parser);
  };

  AsmPrinter& operator<<(
    AsmPrinter& printer, const sbg::Map& m);

  template<>
  struct FieldParser<sbg::OrdDomPWMap>
  {
    static FailureOr<sbg::OrdDomPWMap>
    parse(AsmParser& parser);
  };

  AsmPrinter& operator<<(
      AsmPrinter& printer, const sbg::OrdDomPWMap& m);

  template<>
  struct FieldParser<sbg::PWMap>
  {
    static FailureOr<sbg::PWMap>
    parse(AsmParser& parser);
  };

  AsmPrinter& operator<<(
      AsmPrinter& printer, const sbg::PWMap& m);

  template<>
  struct FieldParser<sbg::EqPath>
  {
    static FailureOr<sbg::EqPath>
    parse(AsmParser& parser);
  };

  AsmPrinter& operator<<(
      AsmPrinter& printer, const sbg::EqPath& eq_path);
}

#endif // MARCO_DIALECTS_SBG_ATTRIBUTES_H
