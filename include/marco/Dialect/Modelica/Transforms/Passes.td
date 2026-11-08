#ifndef MARCO_DIALECT_MODELICA_TRANSFORMS_PASSES_TD
#define MARCO_DIALECT_MODELICA_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def InstantiatePass : Pass<"modelica-instantiate", "mlir::ModuleOp">
{
    let summary = "Instantiate a model.";

    let description = [{
        Build the instance tree of the given class, and
        replace the class-level operations with a `modelica.model` and the
        `modelica.function` operations of the functions it calls.

        Instantiation:
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

def ExpandConnectionsPass
    : Pass<"modelica-expand-connections", "mlir::ModuleOp">
{
    let summary = "Replace connect-equations with connection equations.";

    let description = [{
        Build the connection sets of the instantiated models and replace the
        connect-equations with the equations they imply:
        equalities of the potential variables and zero-sum equations of the
        flow variables, unconnected flow variables being set to zero.

        Connections between arrays of instances are processed element-wise,
        and the generated equations are grouped back into for-equations
        whenever their indices follow regular patterns.
    }];

    let dependentDialects = ["mlir::modelica::ModelicaDialect"];
}

def FlattenPass : Pass<"modelica-flatten", "mlir::ModuleOp">
{
    let summary = "Flatten the instance tree of the models.";

    let description = [{
        Replace the instance tree of the models with their primitive
        variables. Each variable is named after its dotted
        path, and its dimensions are the ones of the enclosing arrays of
        instances followed by its own ones. The component references are
        updated accordingly, and the modifications of the variables are
        extended to the whole arrays.

        The connect-equations must have been already expanded.
    }];

    let dependentDialects = ["mlir::modelica::ModelicaDialect"];
}

#endif // MARCO_DIALECT_MODELICA_TRANSFORMS_PASSES_TD
