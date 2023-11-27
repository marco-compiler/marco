#ifndef MARCO_DIALECTS_SBG_SBG_TD
#define MARCO_DIALECTS_SBG_SBG_TD

include "marco/Dialect/SBG/SBGAttributes.td"
include "marco/Dialect/SBG/SBGInterfaces.td"
include "marco/Dialect/SBG/SBGTypes.td"
include "marco/Dialect/SBG/SBGDialect.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"


//===---------------------------------------------------------------------===//
// SBG_Op class
//===---------------------------------------------------------------------===//

class SBG_Op<string mnemonic, list<Trait> traits = []> :
    Op<SBG_Dialect, mnemonic, traits>;

//===---------------------------------------------------------------------===//
// SBG operations
//===---------------------------------------------------------------------===//

def NodeTypeConstraint
    : Type<CPred<"$_self.isa<::mlir::sbg::NodeType>()">
                 , "node", "::mlir::sbg::NodeType">;

def NodeOp : SBG_Op<"node">
{
    let summary = "Node operation";

    let description = [{
        Create a node represented by an identifier and a set. The first
        specifies the corresponding SBG-vertex, and the second it enumerates
        all of the individual (scalar) vertices.

        Example:

        ```mlir
        %0 = sbg.node {id = 0, elems = #sbg<mdi [1:1:10]x[1:1:10]>} : !sbg.node
        ```
    }];

    let arguments = (ins
        TypedAttrInterface:$id,
        TypedAttrInterface:$elems
    );
    let results = (outs NodeTypeConstraint:$result);

    let assemblyFormat = [{ attr-dict `:` type($result) }];
}

def EdgeTypeConstraint
    : Type<CPred<"$_self.isa<::mlir::sbg::EdgeType>()">
                 , "node", "::mlir::sbg::EdgeType">;

def EdgeOp : SBG_Op<"edge">
{
    let summary = "Edge operation";

    let description = [{
        Create an edge represented by an identifier and two maps. The first
        specifies the corresponding SBG-edge, and second and third are the
        compact representation of individual (scalar) edges.

        //TODO
        Example:

        ```mlir
        ```
    }];

    let arguments = (ins
        TypedAttrInterface:$id,
        TypedAttrInterface:$map1,
        TypedAttrInterface:$map2
    );
    let results = (outs EdgeTypeConstraint:$result);

    let assemblyFormat = [{ attr-dict `:` type($result) }];
}

#endif // MARCO_DIALECTS_SBG_SBG_TD
