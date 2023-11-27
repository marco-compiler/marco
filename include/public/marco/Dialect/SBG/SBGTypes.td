#ifndef MARCO_DIALECTS_SBG_SBGTYPES_TD
#define MARCO_DIALECTS_SBG_SBGTYPES_TD

include "marco/Dialect/SBG/SBGDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===---------------------------------------------------------------------===//
// SBG_Type class
//===---------------------------------------------------------------------===//

// Base class for Modelica dialect types.
class SBG_Type<
    string name,
    list<Trait> traits = [],
    string baseCppClass = "::mlir::Type">
    : TypeDef<SBG_Dialect, name, traits, baseCppClass>
{
    let mnemonic = ?;
}

//===---------------------------------------------------------------------===//
// SBG type definitions
//===---------------------------------------------------------------------===//

def OrdSetType : SBG_Type<"OrdSet">
{
    let summary = "Ordered set of multi-dimensional intervals type";
    let mnemonic = "ord_set";

    let description = [{
        Type representing an ordered container of non repeated
        multi-dimensional intervals.
    }];
}

def SetType : SBG_Type<"Set">
{
    let summary = "Set of multi-dimensional intervals type";
    let mnemonic = "set";

    let description = [{
        Type representing a set of multi-dimensional intervals.
    }];
}

// _________________________________________________________________________ //

def OrdDomPWMapType : SBG_Type<"OrdDomPWMap">
{
    let summary = "Ordered domain piecewise map type";
    let mnemonic = "ord_pw_map";

    let description = [{
        Type representing an unordered collection of maps, where in each
        of them the domain is ordered.
    }];
}

def PWMapType : SBG_Type<"PWMap">
{
    let summary = "Piecewise map type";
    let mnemonic = "pw_map";

    let description = [{
        Type representing an unordered collection of maps.
    }];
}

// _________________________________________________________________________ //

def NodeType : SBG_Type<"Node">
{
    let summary = "Node type";
    let mnemonic = "node";

    let description = [{
        Type representing an identifier and a set. The first specifies the
        corresponding SBG-vertex, and the second it enumerates all of the
        individual (scalar) vertices.
    }];
}

def EdgeType : SBG_Type<"Edge">
{
    let summary = "Edge type";
    let mnemonic = "edge";

    let description = [{
        Type representing an identifier and two maps. The first
        specifies the corresponding SBG-edge, and second and third are the
        compact representation of individual (scalar) edges.
    }];
}

#endif // MARCO_DIALECTS_SBG_SBGTYPES_TD
