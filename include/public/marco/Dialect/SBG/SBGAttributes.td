#ifndef MARCO_DIALECTS_SBG_SBGATTRIBUTES_TD
#define MARCO_DIALECTS_SBG_SBGATTRIBUTES_TD

include "marco/Dialect/SBG/SBGDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===---------------------------------------------------------------------===//
// SBG_Attr class
//===---------------------------------------------------------------------===//

class SBG_Attr<string name, list<Trait> traits = []>
    : AttrDef<SBG_Dialect, name, traits>
{
    let mnemonic = ?;
}

//===---------------------------------------------------------------------===//
// SBG attribute definitions
//===---------------------------------------------------------------------===//

class RationalParameter<string desc>
    : AttrOrTypeParameter<
      "SBG::Util::RATIONAL", desc, "const SBG::Util::RATIONAL &"
    >;

def SBG_RationalAttr : SBG_Attr<"Rational">
{
    let summary = "Rational";

    let description = [{
        Attribute representing a rational using two integers.
    }];

    let mnemonic = "rational";
    let parameters = (ins RationalParameter<"Rational">:$value);
    let assemblyFormat = "$value";
}

// _________________________________________________________________________ //

class IntervalParameter<string desc>
    : AttrOrTypeParameter<
      "SBG::LIB::Interval", desc, "const SBG::LIB::Interval&"
    >;

def SBG_IntervalAttr : SBG_Attr<"Interval">
{
    let summary = "Natural interval";

    let description = [{
        Attribute representing a collection of naturals with a fixed step.
    }];

    let mnemonic = "interval";
    let parameters = (ins IntervalParameter<"Interval">:$value);
    let assemblyFormat = "$value";
}

// _________________________________________________________________________ //

class MDIParameter<string desc>
    : AttrOrTypeParameter<
      "SBG::LIB::MultiDimInter", desc, "const SBG::LIB::MultiDimInter&"
    >;

def SBG_MDIAttr : SBG_Attr<"MDI">
{
    let summary = "Multi dimensional interval";

    let description = [{
        Attribute representing the cartesian product of an ordered collection
        of intervals.
    }];

    let mnemonic = "mdi";
    let parameters = (ins MDIParameter<"MDI">:$value);
    let assemblyFormat = "$value";
}

// _________________________________________________________________________ //

class OrdSetParameter<string desc>
    : AttrOrTypeParameter<"SBG::LIB::OrdSet", desc, "const SBG::LIB::OrdSet&">;

def SBG_OrdSetAttr : SBG_Attr<"OrdSet", [TypedAttrInterface]>
{
    let summary = "Ordered uni-dimensional set";

    let description = [{
        Attribute representing an ordered collection of mdis.
    }];

    let mnemonic = "ord_set";

    let parameters = (ins
        OrdSetParameter<"Ordered set">:$value
        , AttributeSelfTypeParameter<"">:$type
    );

    let assemblyFormat = "$value";
}

// _________________________________________________________________________ //

class SetParameter<string desc>
    : AttrOrTypeParameter<
      "SBG::LIB::UnordSet", desc, "const SBG::LIB::UnordSet&"
    >;

def SBG_SetAttr : SBG_Attr<"Set", [TypedAttrInterface]>
{
    let summary = "Unordered collection of mdis";

    let description = [{
        Attribute representing an unordered collection of multi-dimensional
        mdis.
    }];

    let mnemonic = "set";
    let parameters = (ins
        SetParameter<"Set">:$value
        , AttributeSelfTypeParameter<"">:$type
    );

    let assemblyFormat = "$value";
}

// _________________________________________________________________________ //

class LinearExpParameter<string desc>
    : AttrOrTypeParameter<"SBG::LIB::LExp", desc, "const SBG::LIB::LExp&">;

def SBG_LinearExpAttr : SBG_Attr<"LinearExp">
{
    let summary = "Linear expression";

    let description = [{
        Attribute representing a linear expression of rational numbers.
    }];

    let mnemonic = "linear_exp";
    let parameters = (ins LinearExpParameter<"LExp">:$value);
    let assemblyFormat = "$value";
}

// _________________________________________________________________________ //

class ExpParameter<string desc>
    : AttrOrTypeParameter<"SBG::LIB::Exp", desc, "const SBG::LIB::Exp&">;

def SBG_ExpAttr : SBG_Attr<"Exp">
{
    let summary = "Expression";

    let description = [{
        Attribute representing an ordered collection of linear expressions,
        that is, a multi-dimensional expression.
    }];

    let mnemonic = "exp";
    let parameters = (ins LinearExpParameter<"Exp">:$value);
    let assemblyFormat = "$value";
}

// _________________________________________________________________________ //

class OrdDomMapParameter<string desc>
    : AttrOrTypeParameter<
      "SBG::LIB::CanonMap", desc, "const SBG::LIB::CanonMap&"
    >;

def SBG_OrdDomMapAttr : SBG_Attr<"OrdDomMap">
{
    let summary = "Ordered domain map";

    let description = [{
        Attribute representing domain/expression pair, in which the domain is
        an ordered uni-dimensional set.
    }];

    let mnemonic = "ord_dom_map";
    let parameters = (ins OrdDomMapParameter<"Ordered domain map">:$value);
    let assemblyFormat = "$value";
}

// _________________________________________________________________________ //

class MapParameter<string desc>
    : AttrOrTypeParameter<
      "SBG::LIB::BaseMap", desc, "const SBG::LIB::BaseMap&"
    >;

def SBG_MapAttr : SBG_Attr<"Map">
{
    let summary = "Map";

    let description = [{
        Attribute representing domain/expression pair, in which the domain is
        an unordered collection of mdis.
    }];

    let mnemonic = "map";
    let parameters = (ins MapParameter<"Map">:$value);
    let assemblyFormat = "$value";
}

// _________________________________________________________________________ //

class OrdDomPWMapParameter<string desc>
    : AttrOrTypeParameter<
      "SBG::LIB::CanonPWMap", desc, "const SBG::LIB::CanonPWMap&"
    >;

def SBG_OrdDomPWMapAttr : SBG_Attr<"OrdDomPWMap", [TypedAttrInterface]>
{
    let summary = "Ordered domain piece-wise map";

    let description = [{
        Attribute representing an unordered collection of ordered domain maps.
    }];

    let mnemonic = "ord_dom_pw_map";
    let parameters = (ins
        OrdDomPWMapParameter<"Ordered domain piece-wise map">:$value
        , AttributeSelfTypeParameter<"">:$type
    );
    let assemblyFormat = "$value";
}

// _________________________________________________________________________ //

class PWMapParameter<string desc>
    : AttrOrTypeParameter<
      "SBG::LIB::BasePWMap", desc, "const SBG::LIB::BasePWMap&"
    >;

def SBG_PWMapAttr : SBG_Attr<"PWMap", [TypedAttrInterface]>
{
    let summary = "Piece-wise map";

    let description = [{
        Attribute representing an unordered collection of maps.
    }];

    let mnemonic = "pw_map";
    let parameters = (ins
        PWMapParameter<"PWMap">:$value
        , AttributeSelfTypeParameter<"">:$type
    );
    let assemblyFormat = "$value";
}

#endif // MARCO_DIALECTS_SBG_SBGATTRIBUTES_TD
