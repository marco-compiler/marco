#ifndef MARCO_DIALECT_MODELICA_TRANSFORMS_PASSES_TD
#define MARCO_DIALECT_MODELICA_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def InstantiationPass : Pass<"modelica-instantiation", "mlir::ModuleOp">
{
    let summary = "Instantiate a model.";

    let description = [{
        Build the instance tree of the given class, and
        replace the class-level operations with a `modelica.model` and the
        `modelica.function` operations of the functions it calls.

        Instantiation consists of the following steps:
        - expands the extends-clauses and the short class definitions;
        - merges the modifications, the outer ones overriding the inner ones,
          and splits the ones applied to arrays of
          instances;
        - applies the redeclarations, including class-extends ones;
        - resolves the names according to the lookup rules of the language,
          turning component references into absolute paths, and replacing
          package constants with their values;
        - evaluates the structural parameters: array dimensions, conditions
          of conditional components, and the ranges and
          conditions of the for- and if-equations containing
          connect-equations;
        - resolves outer elements to the corresponding inner ones,
          adding the missing inner ones at the top level;
        - determines whether the connectors of connect-equations are inside
          or outside.
    }];

    let options = [
        Option<"modelName", "model", "std::string", "",
               "Dotted name of the class to be instantiated">
    ];

    let dependentDialects = ["mlir::modelica::ModelicaDialect"];
}

def ConnectionExpansionPass
    : Pass<"modelica-connection-expansion", "mlir::ModuleOp">
{
    let summary = "Replace connect-equations with connection equations.";

    let description = [{
        Build the connection sets of the instantiated models and replace the
        connect-equations with the equations they imply: equalities of the
        potential variables and zero-sum equations of the flow variables,
        unconnected flow variables being set to zero.

        Arrays are never processed element-wise: the connection sets are
        computed on index sets, by means of a graph whose vertices are the
        primitive variables of the connectors and whose edges are the
        families of connections given by the connect-equations, with their
        indices as affine functions of the iterators of the enclosing
        for-equations. The equations are emitted as for-equations over the
        ranges of the resulting index sets.
    }];

    let dependentDialects = ["mlir::modelica::ModelicaDialect"];
}

def FlatteningPass : Pass<"modelica-flattening", "mlir::ModuleOp">
{
    let summary = "Flatten the instance tree of the models.";

    let description = [{
        Replace the instance tree of the models with their primitive variables.
        Each variable is named after its dotted path, and its dimensions are
        the ones of the enclosing arrays of instances followed by its own ones.
        The component references are updated accordingly, and the modifications
        of the variables are extended to the whole arrays.
    }];

    let dependentDialects = ["mlir::modelica::ModelicaDialect"];
}

#endif // MARCO_DIALECT_MODELICA_TRANSFORMS_PASSES_TD
