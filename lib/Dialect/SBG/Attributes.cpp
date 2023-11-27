#include "marco/Dialect/SBG/Attributes.h"
#include "marco/Dialect/SBG/SBGDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::sbg;

//===---------------------------------------------------------------------===//
// Tablegen attribute definitions
//===---------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/SBG/SBGAttributes.cpp.inc"

//===---------------------------------------------------------------------===//
// SBGDialect
//===---------------------------------------------------------------------===//

namespace mlir::sbg
{
  void SBGDialect::registerAttributes()
  {
    addAttributes<
      #define GET_ATTRDEF_LIST
      #include "marco/Dialect/SBG/SBGAttributes.cpp.inc"
    >();
  }
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

namespace mlir
{
  FailureOr<Rational> FieldParser<Rational>::parse(AsmParser& parser)
  {
    uint64_t num, den;

    if (parser.parseInteger(num)) {
      return failure();
    }

    if (succeeded(parser.parseOptionalKeyword("/"))) {
      if (parser.parseInteger(den)) {
        return failure();
      }
    }

    return sbg::Rational(num, den);
  }

  AsmPrinter& operator<<(AsmPrinter& printer, const Rational& r)
  {
    uint64_t num = r.numerator(), den = r.denominator();

    if (num == 0){
      printer << "0";
      return printer;
    }

    printer << num;
    if (den != 1) {
      printer << "/" << den;
    }

    return printer;
  }

  FailureOr<std::optional<Rational>>
  FieldParser<std::optional<Rational>>::parse(AsmParser& parser)
  {
    uint64_t num, den;

    if (failed(parser.parseInteger(num))) {
      return std::optional<Rational>(std::nullopt);
    }

    if (succeeded(parser.parseOptionalKeyword("/"))) {
      if (parser.parseInteger(den)) {
        return failure();
      }
    }

    return std::optional<Rational>(sbg::Rational(num, den));
  }

  AsmPrinter& operator<<(AsmPrinter& printer, const std::optional<Rational>& r)
  {
    if (r) {
      printer << r;
    }

    return printer;
  }

  FailureOr<Interval> FieldParser<Interval>::parse(AsmParser& parser)
  {
    uint64_t begin, step, end;

    if (parser.parseLSquare()
        || parser.parseInteger(begin)
        || parser.parseColon()
        || parser.parseInteger(step)
        || parser.parseColon()
        || parser.parseInteger(end)
        || parser.parseRSquare()) {
      return failure();
    }

    return sbg::Interval(begin, step, end);
  }

  AsmPrinter& operator<<(AsmPrinter& printer, const Interval& i)
  {
    printer << "[" << i.begin() << ":" << i.step() << ":" << i.end() << "]";

    return printer;
  }

  FailureOr<MDI> FieldParser<MDI>::parse(AsmParser& parser)
  {
    MDI result;

    do {
      FailureOr<Interval> i = FieldParser<Interval>::parse(parser);

      if (failed(i)) {
        return failure();
      }

      result.emplaceBack(*i);
    } while (succeeded(parser.parseOptionalKeyword("x")));

    return result;
  }

  AsmPrinter& operator<<(AsmPrinter& printer, const MDI& mdi)
  {
    unsigned int sz = mdi.size();

    if (sz > 0) {
      for (unsigned int j = 0; j < sz - 1; ++j) {
        printer << mdi[j] << "x";
      }
      printer << mdi[sz - 1];
    }

    return printer;
  }

  FailureOr<OrdSet> FieldParser<OrdSet>::parse(AsmParser& parser)
  {
    OrdSet result;

    if (parser.parseLBrace()) {
      return failure();
    }

    if (succeeded(parser.parseOptionalRBrace())) {
      return result;
    }

    do {
      FailureOr<MDI> mdi = FieldParser<MDI>::parse(parser);

      if (failed(mdi)) {
        return failure();
      }

      result.emplaceBack(*mdi);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRBrace()) {
      return failure();
    }

    return result;
  }

  AsmPrinter& operator<<(AsmPrinter& printer, const OrdSet& ord_set)
  {
    unsigned int sz = ord_set.size();

    printer << "{";
    if (sz > 0) {
      auto it = ord_set.begin();
      for (unsigned int j = 0; j < sz - 1; ++j) {
        printer << *it << ", ";
        ++it;
      }
      printer << *it;
    }
    printer << "}";

    return printer;
  }

  FailureOr<Set> FieldParser<Set>::parse(AsmParser& parser)
  {
    Set result;

    if (parser.parseLBrace()) {
      return failure();
    }

    if (succeeded(parser.parseOptionalRBrace())) {
      return result;
    }

    do {
      FailureOr<MDI> mdi = FieldParser<MDI>::parse(parser);

      if (failed(mdi)) {
        return failure();
      }

      result.emplaceBack(*mdi);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRBrace()) {
      return failure();
    }

    return result;
  }

  AsmPrinter& operator<<(AsmPrinter& printer, const Set& unord_set)
  {
    unsigned int sz = unord_set.size();

    printer << "{";
    if (sz > 0) {
      auto it = unord_set.begin();
      for (unsigned int j = 0; j < sz - 1; ++j) {
        printer << *it << ", ";
        ++it;
      }
      printer << *it;
    }
    printer << "}";

    return printer;
  }

  FailureOr<LinearExp> FieldParser<LinearExp>::parse(AsmParser& parser)
  {
    Rational m = 1, h = 0;

    FailureOr<std::optional<Rational>> optm
        = FieldParser<std::optional<Rational>>::parse(parser);
    if (succeeded(optm)) {
      m = **optm;
      if (parser.parseKeyword("*x")) {
        return failure();
      }

      if (succeeded(parser.parseOptionalPlus())) {
        FailureOr<std::optional<Rational>> opth
            = FieldParser<std::optional<Rational>>::parse(parser);
        if (failed(opth)) {
          return failure();
        }
        h = **opth;
      }
    }

    else if (succeeded(parser.parseOptionalKeyword("x"))) {
      if (succeeded(parser.parseOptionalPlus())) {
        FailureOr<std::optional<Rational>> opth
            = FieldParser<std::optional<Rational>>::parse(parser);
        if (failed(opth)) {
          return failure();
        }
        h = **opth;
      }
    }

    else {
      FailureOr<std::optional<Rational>> opth
          = FieldParser<std::optional<Rational>>::parse(parser);
      if (failed(opth)) {
        return failure();
      }
      h = **opth;
    }

    return LinearExp(m, h);
  }

  AsmPrinter& operator<<(AsmPrinter& printer, const LinearExp& le)
  {
    Rational slo = le.slope(), off = le.offset();

    if (slo != 0 && slo != 1) {
      if (slo.numerator() != 1) {
        printer << slo.numerator();
      }

      if (slo.denominator() != 1) {
        printer << "x/" << slo.denominator();
      }
      else {
        printer << "x";
      }
    }

    if (slo == 1) {
      printer << "x";
    }

    if (off != 0) {
      if (off > 0 && slo != 0) {
        printer << "+" << off;
      }
      else {
        printer << off;
      }
    }

    if (slo == 0 && off == 0) {
      printer << "0";
    }

    return printer;
  }

  FailureOr<Exp> FieldParser<Exp>::parse(AsmParser& parser)
  {
    Exp result;

    do {
      FailureOr<LinearExp> le = FieldParser<LinearExp>::parse(parser);

      if (failed(le)) {
        return failure();
      }

      result.emplaceBack(*le);
    } while (succeeded(parser.parseOptionalKeyword("|")));

    return result;
  }

  AsmPrinter& operator<<(AsmPrinter& printer, const Exp& le)
  {
    unsigned int sz = le.size();

    if (sz > 0) {
      for (unsigned int j = 0; j < sz-1; ++j) {
        printer << le[j] << "|";
      }
      printer << le[sz-1];
    }

    return printer;
  }

  FailureOr<OrdDomMap> FieldParser<OrdDomMap>::parse(AsmParser& parser)
  {
    FailureOr<OrdSet> s = FieldParser<OrdSet>::parse(parser);
    FailureOr<Exp> e = FieldParser<Exp>::parse(parser);
    if (failed(s)
        || parser.parseKeyword("->")
        || failed(e)) {
      return failure();
    }

    return OrdDomMap(*s, *e);
  }

  AsmPrinter& operator<<(AsmPrinter& printer, const OrdDomMap& m)
  {
    printer << m.dom() << " ↦ " << m.exp();

    return printer;
  }

  FailureOr<Map> FieldParser<Map>::parse(AsmParser& parser)
  {
    FailureOr<Set> s = FieldParser<Set>::parse(parser);
    FailureOr<Exp> e = FieldParser<Exp>::parse(parser);
    if (failed(s)
        || parser.parseKeyword("->")
        || failed(e)) {
      return failure();
    }

    return Map(*s, *e);
  }

  AsmPrinter& operator<<(AsmPrinter& printer, const Map& m)
  {
    printer << m.dom() << " ↦ " << m.exp();

    return printer;
  }

  FailureOr<OrdDomPWMap> FieldParser<OrdDomPWMap>::parse(AsmParser& parser)
  {
    OrdDomPWMap result;

    if (parser.parseKeyword("<<")) {
      return failure();
    }

    if (succeeded(parser.parseOptionalKeyword(">>"))) {
      return result;
    }

    do {
      FailureOr<OrdDomMap> m = FieldParser<OrdDomMap >::parse(parser);

      if (failed(m)) {
        return failure();
      }

      result.emplaceBack(*m);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseKeyword(">>")) {
      return failure();
    }

    return result;
  }

  AsmPrinter& operator<<(AsmPrinter& printer, const OrdDomPWMap& pw)
  {
    unsigned int sz = pw.size();

    printer << "<<";
    if (sz > 0) {
      auto it = pw.begin();
      for (unsigned int j = 0; j < sz - 1; ++j) {
        printer << *it << ",";
        ++it;
      }
      printer << *it;
    }
    printer << ">>";

    return printer;
  }

  FailureOr<PWMap> FieldParser<PWMap>::parse(AsmParser& parser)
  {
    PWMap result;

    if (parser.parseKeyword("<<")) {
      return failure();
    }

    if (succeeded(parser.parseOptionalKeyword(">>"))) {
      return result;
    }

    do {
      FailureOr<Map> m = FieldParser<Map >::parse(parser);

      if (failed(m)) {
        return failure();
      }

      result.emplaceBack(*m);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseKeyword(">>")) {
      return failure();
    }

    return result;
  }

  AsmPrinter& operator<<(AsmPrinter& printer, const PWMap& pw)
  {
    unsigned int sz = pw.size();

    printer << "<<";
    if (sz > 0) {
      auto it = pw.begin();
      for (unsigned int j = 0; j < sz - 1; ++j) {
        printer << *it << ",";
        ++it;
      }
      printer << *it;
    }
    printer << ">>";

    return printer;
  }
} // namespace mlir