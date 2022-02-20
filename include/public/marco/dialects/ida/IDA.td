#ifndef MARCO_DIALECTS_IDA_IDA_TD
#define MARCO_DIALECTS_IDA_IDA_TD

include "marco/dialects/ida/IDAAttributes.td"
include "marco/dialects/ida/IDADialect.td"
include "marco/dialects/ida/IDATypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

// IDA instance type
def IDAInstance : Type<CPred<"$_self.isa<::mlir::ida::InstanceType>()">,
                       "instance", "::mlir::ida::InstanceType">;

// IDA variables list type
def IDAVariables : Type<CPred<"$_self.isa<::mlir::ida::VariablesType>()">,
                               "variables", "::mlir::ida::VariablesType">;

// IDA equation type
def IDAEquation : Type<CPred<"$_self.isa<::mlir::ida::EquationType>()">,
                             "equation", "::mlir::ida::EquationType">;

// IDA variable type
def IDAVariable : Type<CPred<"$_self.isa<::mlir::ida::VariableType>()">,
                             "variable", "::mlir::ida::VariableType">;

// IDA residual function type
def IDAResidualFunction : Type<CPred<"$_self.isa<::mlir::ida::ResidualFunctionType>()">,
                               "residual_function", "::mlir::ida::ResidualFunctionType">;

// IDA Jacobian function type
def IDAJacobianFunction : Type<CPred<"$_self.isa<::mlir::ida::JacobianFunctionType>()">,
                               "jacobian_function", "::mlir::ida::JacobianFunctionType">;

//===----------------------------------------------------------------------===//
// IDA operation definitions
//===----------------------------------------------------------------------===//

class IDA_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<IDA_Dialect, mnemonic, traits>;

def IDA_CreateOp : IDA_Op<"create",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Create an IDA instance.";

    let description = [{
        The  operation takes three operands and returns a new
        IDA instance.
        The operands consist in the number of scalar equations,
        the number of vector equations and the number of array
        variables. The number of scalar variables is not needed
        as it matches by construction the number of scalar equation,
        because each scalar equation is indeed matched to a scalar
        variable.
        The underlying data structures are allocated but not
        initialized; initialization is delegated to the InitOp
        operation, which has to be called after all the data
        regarding equations and variables have been set by means
        of the appropriate operations (i.e. AddEquationOp, etc.).

        Example:

        ```mlir
        %0 = ida.create {scalarEquations = 23; vectorEquations = 7; vectorVariables = 3 } : !ida.instance
        ```
    }];

    let arguments = (ins
        I64Attr:$scalarEquations,
        I64Attr:$vectorEquations,
        I64Attr:$vectorVariables);

    let results = (outs IDAInstance:$instance);

    let assemblyFormat = [{
        attr-dict `:` type($instance)
    }];
}

def IDA_SetStartTimeOp : IDA_Op<"set_start_time",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Set the start time of the simulation.";

    let description = [{
        The operation sets the start time to be used by
        IDA during the simulation.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        ida.set_start_time %0 { time = 0.0 } : !ida.instance
        ``
    }];

    let arguments = (ins
        IDAInstance:$instance,
        F64Attr:$time);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_SetEndTimeOp : IDA_Op<"set_end_time",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Set the end time of the simulation.";

    let description = [{
        The operation sets the end time to be used by
        IDA during the simulation.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        ida.set_end_time %0 { time = 0.0 } : !ida.instance
        ``
    }];

    let arguments = (ins
        IDAInstance:$instance,
        F64Attr:$time);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_SetRelativeToleranceOp : IDA_Op<"set_relative_tolerance",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Set the relative tolerance.";

    let description = [{
        Set the relative tolerance to be used during the simulation.

        Relative tolerance is intended as the difference between the
        values computed through the n-th and the (n+1)-th order BDF
        method, divided by the absolute value given by the (n+1)-th
        order BDF method.

        It is mandatory to set the parameter higher than the minimum
        precision of the floating point unit roundoff (10^-15 for
        doubles).

        It is also highly suggested to set the parameter lower than
        10^-3 in order to avoid inaccurate results. IDA defaults to
        10^-6.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        ida.set_relative_tolerance %0 { tolerance = 0.000001 } : !ida.instance
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        F64Attr:$tolerance);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_SetAbsoluteToleranceOp : IDA_Op<"set_absolute_tolerance",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Set the absolute tolerance.";

    let description = [{
        Set the absolute tolerance to be used during the simulation.

        Absolute tolerance is intended as the maximum acceptable
        difference between the values computed through the n-th and
        the (n+1)-th order BDF method.

        Absolute tolerance is used to substitute relative tolerance
        when the value converges to zero. When this happens, in fact,
        the relative error would tend to infinity, thus exceeding the
        set tolerance.

        It is mandatory to set the parameter higher than the minimum
        precision of the floating point unit roundoff (10^-15 for
        doubles).

        It is also highly suggested to set the parameter lower than
        10^-3 in order to avoid inaccurate results. IDA defaults to
        10^-6.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        ida.set_absolute_tolerance %0 { tolerance = 0.000001 } : !ida.instance
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        F64Attr:$tolerance);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_GetCurrentTimeOp : IDA_Op<"get_current_time",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Get the time reached by IDA.";

    let description = [{
        The operation returns the time reached by the simulation.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        %1 = ida.get_current_time %0 : !ida.instance -> f64
        ```
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs F64:$time);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($time)
    }];
}

def IDA_AddEquationOp : IDA_Op<"add_equation",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the information regarding an equation iteration range.";

    let description = [{
        The operation is intended to inform IDA about the iteration
        ranges of an equation.

        The ranges are expressed as integer pairs of values, where
        each pair consists in the beginning and ending iteration value,
        with the former being strictly less than the latter.

        For example, an equation iterating on two indices `i`, `j` and `k`,
        with `i` belonging to `[4,7)`, `j` belonging to `[9, 13)` and `k`
        belonging to `[1, 3)`, would be represented by the following 2-D
        2x3 array: `[[4, 7], [9, 13], [1, 3]]`.

        The operation returns a reference to the equation added to the
        IDA instance.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        %1 = ... : memref<2x?xi64>
        ida.add_equation %0, %1 : (ida.instance, memref<2x?xi64>) -> ida.equation
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        I64MemRef:$equationRanges);

    let results = (outs IDAEquation:$equation);

    let assemblyFormat = [{
        $instance `,` $equationRanges attr-dict `:` `(` type($instance) `,` type($equationRanges) `)` `->` type($equation)
    }];
}

def IDA_AddVariableOp : IDA_Op<"add_variable",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the information regarding the dimensions of a variable.";

    let description = [{
        The operation is intended to inform IDA about the dimensions of
        a variable.

        The operation returns a reference to the variable added to the
        IDA instance.

        The `state` attribute is used to tell IDA whether the variables
        is a state one.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        %1 = ... : memref<?xi64>
        ida.add_variable %0, %1 { state = true } : (ida.instance, memref<?xi64>) -> ida.variable
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        I64MemRef:$arrayDimensions,
        I1Attr:$state);

    let results = (outs IDAVariable:$variable);

    let assemblyFormat = [{
        $instance `,` $arrayDimensions attr-dict `:` `(` type($instance) `,` type($arrayDimensions) `)` `->` type($variable)
    }];
}

def IDA_AddVariableAccessOp : IDA_Op<"add_variable_access",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the information regarding the access to an array variable inside an equation.";

    let description = [{
        The operation is used to inform IDA about how an equation accesses
        an array variable.

        ```mlir
        %0 = ... : !ida.instance
        %1 = ... : !ida.equation
        %2 = ... : !ida.variable
        // TODO
        ida.add_variable_access %0, %1, %2 { access = affine_map<(d0, d1)[s0] -> (d0, d0 + d1 + s0 floordiv 2)> } : !ida.instance, !ida.equation, !ida.variable
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        IDAEquation:$equation,
        IDAVariable:$variable,
        AffineMapAttr:$access);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `,` type($equation) `,` type($variable)
    }];
}

def IDA_GetVariableOp : IDA_Op<"get_variable",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins
        IDAInstance:$instance,
        IDAVariable:$variable);

    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $instance `,` $variable attr-dict `:` `(` type($instance) `,` type($variable) `)` `->` type($result)
    }];
}

def IDA_GetVariablesOp : IDA_Op<"get_variables",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Get the list of scalar variables belonging to IDA.";

    let description = [{
        Get the full list of the variables that have been added
        to the IDA instance through the `ida.add_variable` operation.
        The list is composed of array variables whose elements are
        placed contiguously.

        The `ida.get_variable` operation can be used to
        obtain a reference to a specific array variable living
        inside this list.

        ```mlir
        %0 = ... : !ida.instance
        %1 = ida.get_variables %0 : !ida.instance -> !ida.variables
        ```
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs IDAVariables:$variables);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($variables)
    }];
}

def IDA_GetDerivativesOp : IDA_Op<"get_derivatives",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Get the list of scalar derivatives belonging to IDA.";

    let description = [{
        Get the full list of the derivatives that have been added
        to the IDA instance through the `ida.add_variable` operation.
        The list is composed of array variables whose elements are
        placed contiguously.

        The `ida.get_variable` operation can be used to
        obtain a reference to a specific array derivative living
        inside this list. Note that the same `ida.get_variable`
        operation is used for both variables and derivatives,
        because derivatives are conceptually variables.

        ```mlir
        %0 = ... : !ida.instance
        %1 = ida.get_derivatives %0 : !ida.instance -> !ida.variables
        ```
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs IDAVariables:$ders);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($ders)
    }];
}

def IDA_ResidualFunctionOp : IDA_Op<"residual_function",
    [IsolatedFromAbove, FunctionLike, AutomaticAllocationScope, Symbol]>
{
    let summary = "Residual function of a vector equation.";

    let description = [{
        The residual function of an equation is a function that computes
        the residual error of that equation.

        The residual error of an equation consists in the difference
        between its right-hand side and left-hand side values.

        The function takes four parameters:
          1. the current time of the simulation.
          2. the indices of the scalar equation to be examined.
          3. the variables list.
          4. the derivatives list.

        Example:

        ```mlir
        ida.residual_function @foo {
            ^bbo(%time : f64, %indices : memref<?xi64>, %vars : !ida.variables, %ders : !ida.variables) {
                ...
                %diff = ... : f64
                ida.return %diff : f64
            }
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 4> args;
            args.push_back($_builder.getF64Type());

            llvm::SmallVector<int64_t, 1> equationIndicesShape;
            equationIndicesShape.push_back(-1);

            auto equationIndicesType = MemRefType::get(equationIndicesShape, $_builder.getIndexType());
            args.push_back(equationIndicesType);

            args.push_back(VariablesType::get($_builder.getContext()));
            args.push_back(VariablesType::get($_builder.getContext()));

            auto functionType = $_builder.getFunctionType(args, $_builder.getF64Type());
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr);
        }]>
    ];

    let extraClassDeclaration = [{
        public:
            FunctionType getType() {
                return getTypeAttr().getValue().cast<FunctionType>();
            }

            //===------------------------------------------------------------------===//
            // FunctionOpInterface Methods
            //===------------------------------------------------------------------===//

            /// Returns the argument types of this function.
            ArrayRef<Type> getArgumentTypes() {
                return getType().getInputs();
            }

            /// Returns the result types of this function.
            ArrayRef<Type> getResultTypes() {
                return getType().getResults();
            }

            //===------------------------------------------------------------------===//
            // SymbolOpInterface Methods
            //===------------------------------------------------------------------===//

            bool isDeclaration() {
                return isExternal();
            }

        private:
            // This trait needs access to the hooks defined below.
            friend class OpTrait::FunctionLike<ResidualFunctionOp>;

            /// Returns the number of arguments. This is a hook for
            /// OpTrait::FunctionLike.
            unsigned getNumFuncArguments() {
                return getType().getInputs().size();
            }

            /// Returns the number of results. This is a hook for OpTrait::FunctionLike.
            unsigned getNumFuncResults() {
                return getType().getResults().size();
            }

            /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
            /// attribute is present and checks if it holds a function type. Ensures
            /// getType, getNumFuncArguments, and getNumFuncResults can be called
            /// safely.
            LogicalResult verifyType() {
                auto type = getTypeAttr().getValue();

                if (!type.isa<FunctionType>()) {
                    return emitOpError("requires '" + getTypeAttrName() + "' attribute of function type");
               }

                return success();
            }
    }];
}

def IDA_JacobianFunctionOp : IDA_Op<"jacobian_function",
    [IsolatedFromAbove, FunctionLike, AutomaticAllocationScope, Symbol]>
{
    let summary = "Jacobian function of a vector equation.";

    let description = [{
        The Jacobian function of an equation is a function that computes
        an element of the Jacobian matrix of that equation.

        The function takes seven parameters:
          1. the current time of the simulation.
          2. the indices of the scalar equation to be examined.
          3. the variables list.
          4. the derivatives list.
          5. the vector variable belonging to the IDA instance.
          6. the indices of the scalar variable with respect to which
             the equation has to be derived.
          7. the IDA alpha parameter.

        Example:

        ```mlir
        init {
          %pippo = ida.add_variable %x : !ida.variable
        }

        map.insert(x -> pippo)

        modelica.equation {
            %x
        }

        ida.jacobian_function @foo {
            ^bbo(%time : f64, %equationIndices : memref<?xi64>, %vars : !ida.variables, %ders : !ida.variables, %var : !ida.variable, %variableIndices : memref<?xi64>, %alpha : f64) {
                ida.get_var %instance, %pippo : !modelica.array<4x!modelica.int>
                ...
                %diff = ... : f64
                ida.return %diff : f64
            }
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 6> args;
            args.push_back($_builder.getF64Type());

            llvm::SmallVector<int64_t, 1> equationIndicesShape;
            equationIndicesShape.push_back(-1);

            auto equationIndicesType = MemRefType::get(equationIndicesShape, $_builder.getIndexType());
            args.push_back(equationIndicesType);

            args.push_back(VariablesType::get($_builder.getContext()));
            args.push_back(VariablesType::get($_builder.getContext()));
            args.push_back(VariableType::get($_builder.getContext()));

            llvm::SmallVector<int64_t, 1> variableIndicesShape;
            variableIndicesShape.push_back(-1);

            auto variableIndicesType = MemRefType::get(variableIndicesShape, $_builder.getIndexType());
            args.push_back(variableIndicesType);

            args.push_back($_builder.getF64Type());

            auto functionType = $_builder.getFunctionType(args, $_builder.getF64Type());
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr);
        }]>
    ];

    let extraClassDeclaration = [{
        public:
            FunctionType getType() {
                return getTypeAttr().getValue().cast<FunctionType>();
            }

            //===------------------------------------------------------------------===//
            // FunctionOpInterface Methods
            //===------------------------------------------------------------------===//

            /// Returns the argument types of this function.
            ArrayRef<Type> getArgumentTypes() {
                return getType().getInputs();
            }

            /// Returns the result types of this function.
            ArrayRef<Type> getResultTypes() {
                return getType().getResults();
            }

            //===------------------------------------------------------------------===//
            // SymbolOpInterface Methods
            //===------------------------------------------------------------------===//

            bool isDeclaration() {
                return isExternal();
            }

        private:
            // This trait needs access to the hooks defined below.
            friend class OpTrait::FunctionLike<JacobianFunctionOp>;

            /// Returns the number of arguments. This is a hook for
            /// OpTrait::FunctionLike.
            unsigned getNumFuncArguments() {
                return getType().getInputs().size();
            }

            /// Returns the number of results. This is a hook for OpTrait::FunctionLike.
            unsigned getNumFuncResults() {
                return getType().getResults().size();
            }

            /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
            /// attribute is present and checks if it holds a function type. Ensures
            /// getType, getNumFuncArguments, and getNumFuncResults can be called
            /// safely.
            LogicalResult verifyType() {
                auto type = getTypeAttr().getValue();

                if (!type.isa<FunctionType>()) {
                    return emitOpError("requires '" + getTypeAttrName() + "' attribute of function type");
               }

                return success();
            }
    }];
}

def IDA_ReturnOp : IDA_Op<"return",
    [NoSideEffect,
     ParentOneOf<["ResidualFunctionOp", "JacobianFunctionOp"]>,
     ReturnLike, Terminator]>
{
    let summary = "Return operation.";

    let description = [{
        The operation represents a return operation within a function.
        The operation takes variable number of operands and produces no results.
        The operand number and types must match the signature of the function
        that contains the operation.
    }];

    let arguments = (ins Variadic<AnyType>:$operands);

    let builders = [
        OpBuilder<(ins), [{
            build($_builder, $_state, llvm::None);
        }]>];

    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def IDA_AddResidualOp : IDA_Op<"add_residual",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the residual function of an equation.";

    let description = [{
        The operation is intended to inform IDA about the residual
        function to be used for an equation already declared inside
        IDA.

        ```mlir
        ida.residual_function @foo {
            ...
        }

        %0 = ... : !ida.instance
        %1 = ... : !ida.equation
        ida.add_residual %0, %1 { function : "foo" } : !ida.instance, !ida.equation
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        IDAEquation:$equation,
        SymbolNameAttr:$function);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation attr-dict `:` type($instance) `,` type($equation)
    }];
}

def IDA_AddJacobianOp : IDA_Op<"add_jacobian",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the Jacobian function of an equation.";

    let description = [{
        The operation is intended to inform IDA about the Jacobian
        function to be used for an equation already declared inside
        IDA.

        ```mlir
        ida.jacobian_function @foo {
            ...
        }

        %0 = ... : !ida.instance
        %1 = ... : !ida.equation
        ida.add_jacobian %0, %1 { function : "foo" } : !ida.instance, !ida.equation
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        IDAEquation:$equation,
        SymbolNameAttr:$function);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation attr-dict `:` type($instance) `,` type($equation)
    }];
}

def IDA_InitOp : IDA_Op<"init",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Initialize IDA environment.";

    let description = [{
        The operation initializes the IDA environment.
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs I1:$result);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($result)
    }];
}

def IDA_StepOp : IDA_Op<"step",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins
        IDAInstance:$instance,
        OptionalAttr<F64Attr>:$timeStep);

    let results = (outs I1:$result);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($result)
    }];
}

def IDA_FreeOp : IDA_Op<"free",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs I1:$result);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($result)
    }];
}

def IDA_PrintStatisticsOp : IDA_Op<"print_statistics",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

#endif // MARCO_DIALECTS_IDA_IDA_TD
