#ifndef MARCO_DIALECTS_IDA_IDA_TD
#define MARCO_DIALECTS_IDA_IDA_TD

include "marco/Dialect/IDA/IDAAttributes.td"
include "marco/Dialect/IDA/IDADialect.td"
include "marco/Dialect/IDA/IDATypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

// IDA instance type
def IDAInstance : Type<CPred<"$_self.isa<::mlir::ida::InstanceType>()">,
                       "instance", "::mlir::ida::InstanceType">;

// IDA equation type
def IDAEquation : Type<CPred<"$_self.isa<::mlir::ida::EquationType>()">,
                             "equation", "::mlir::ida::EquationType">;

// IDA variable type
def IDAVariable : Type<CPred<"$_self.isa<::mlir::ida::VariableType>()">,
                             "variable", "::mlir::ida::VariableType">;

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
        // TODO aggiornare
        The operand consists in the number of total amount of
        scalar equations of the model. The number of scalar variables
        is not needed as it matches by construction the number of
        scalar equation, because each scalar equation is indeed
        matched to a scalar variable.
        The underlying data structures are allocated but not
        initialized; initialization is delegated to the InitOp
        operation, which has to be called after all the data
        regarding equations and variables have been set by means
        of the appropriate operations (i.e. AddEquationOp, etc.).

        Example:

        ```mlir
        %0 = ida.create { scalarEquations = 23 } : !ida.instance
        ```
    }];

    let arguments = (ins I64Attr:$scalarEquations);

    let results = (outs IDAInstance:$instance);

    let builders = [
        OpBuilder<(ins "IntegerAttr":$scalarEquations), [{
            auto resultType = ida::InstanceType::get($_builder.getContext());
            build($_builder, $_state, resultType, scalarEquations);
        }]>
    ];

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

def IDA_SetTimeStepOp : IDA_Op<"set_time_step",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "";

    let description = [{

    }];

    let arguments = (ins
        IDAInstance:$instance,
        F64Attr:$timeStep);

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
    let results = (outs AnyType:$time);

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
        ArrayAttr:$equationRanges);

    let results = (outs IDAEquation:$equation);

    let builders = [
        OpBuilder<(ins "Value":$idaInstance, "ArrayAttr":$equationRanges), [{
            auto resultType = ida::EquationType::get($_builder.getContext());
            build($_builder, $_state, resultType, idaInstance, equationRanges);
        }]>
    ];

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($equation)
    }];
}

def IDA_AddAlgebraicVariableOp : IDA_Op<"add_algebraic_variable",
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
        AnyType:$variable,
        I64ArrayAttr:$arrayDimensions,
        SymbolRefAttr:$getter,
        SymbolRefAttr:$setter);

    let results = (outs IDAVariable:$idaVariable);

    let builders = [
        OpBuilder<(ins "Value":$idaInstance, "Value":$variable, "ArrayAttr":$arrayDimensions, "SymbolRefAttr":$getter, "SymbolRefAttr":$setter), [{
            auto resultType = ida::VariableType::get($_builder.getContext());
            build($_builder, $_state, resultType, idaInstance, variable, arrayDimensions, getter, setter);
        }]>,
        OpBuilder<(ins "Value":$idaInstance, "Value":$variable, "ArrayAttr":$arrayDimensions, "StringRef":$getter, "StringRef":$setter), [{
            auto resultType = ida::VariableType::get($_builder.getContext());
            auto getterAttr = $_builder.getSymbolRefAttr(getter);
            auto setterAttr = $_builder.getSymbolRefAttr(setter);
            build($_builder, $_state, resultType, idaInstance, variable, arrayDimensions, getterAttr, setterAttr);
        }]>
    ];

    let assemblyFormat = [{
        $instance `,` $variable attr-dict `:` `(` type($instance) `,` type($variable) `)` `->` type($idaVariable)
    }];
}

def IDA_AddStateVariableOp : IDA_Op<"add_state_variable",
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
        AnyType:$variable,
        I64ArrayAttr:$arrayDimensions,
        SymbolRefAttr:$getter,
        SymbolRefAttr:$setter);

    let results = (outs IDAVariable:$idaVariable);

    let builders = [
        OpBuilder<(ins "Value":$idaInstance, "Value":$variable, "ArrayAttr":$arrayDimensions, "SymbolRefAttr":$getter, "SymbolRefAttr":$setter), [{
            auto resultType = ida::VariableType::get($_builder.getContext());
            build($_builder, $_state, resultType, idaInstance, variable, arrayDimensions, getter, setter);
        }]>,
        OpBuilder<(ins "Value":$idaInstance, "Value":$variable, "ArrayAttr":$arrayDimensions, "StringRef":$getter, "StringRef":$setter), [{
            auto resultType = ida::VariableType::get($_builder.getContext());
            auto getterAttr = $_builder.getSymbolRefAttr(getter);
            auto setterAttr = $_builder.getSymbolRefAttr(setter);
            build($_builder, $_state, resultType, idaInstance, variable, arrayDimensions, getterAttr, setterAttr);
        }]>
    ];

    let assemblyFormat = [{
        $instance `,` $variable attr-dict `:` `(` type($instance) `,` type($variable) `)` `->` type($idaVariable)
    }];
}

def IDA_SetDerivativeOp : IDA_Op<"set_derivative",
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
        IDAVariable:$stateVariable,
        AnyType:$derivative,
        SymbolRefAttr:$getter,
        SymbolRefAttr:$setter);

    let results = (outs);

    let builders = [
        OpBuilder<(ins "Value":$idaInstance, "Value":$stateVariable, "Value":$derivative, "StringRef":$getter, "StringRef":$setter), [{
            auto getterAttr = $_builder.getSymbolRefAttr(getter);
            auto setterAttr = $_builder.getSymbolRefAttr(setter);
            build($_builder, $_state, idaInstance, stateVariable, derivative, getterAttr, setterAttr);
        }]>
    ];

    let assemblyFormat = [{
        $instance `,` $stateVariable `,` $derivative attr-dict `:` type($instance) `,` type($stateVariable) `,` type($derivative)
    }];
}

def IDA_VariableGetterOp : IDA_Op<"variable_getter",
    [IsolatedFromAbove, FunctionLike, AutomaticAllocationScope, Symbol]>
{
    let summary = "";

    let description = [{

    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name, "Type":$resultType, "Type":$variableType, "unsigned int":$variableRank), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 2> args;

            args.push_back(variableType);
            args.append(variableRank, $_builder.getIndexType());

            auto functionType = $_builder.getFunctionType(args, resultType);
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr);
        }]>
    ];

    let extraClassDeclaration = [{
        public:
            llvm::StringRef name() {
                return sym_name();
            }

            FunctionType getType() {
                return getTypeAttr().getValue().cast<FunctionType>();
            }

            BlockArgument getVariable();
            ArrayRef<BlockArgument> getVariableIndices();

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
            friend class OpTrait::FunctionLike<VariableGetterOp>;

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

def IDA_VariableSetterOp : IDA_Op<"variable_setter",
    [IsolatedFromAbove, FunctionLike, AutomaticAllocationScope, Symbol]>
{
    let summary = "";

    let description = [{

    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name, "Type":$variableType, "Type":$valueType, "unsigned int":$variableRank), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 3> args;

            args.push_back(variableType);
            args.push_back(valueType);
            args.append(variableRank, $_builder.getIndexType());

            auto functionType = $_builder.getFunctionType(args, llvm::None);
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr);
        }]>
    ];

    let extraClassDeclaration = [{
        public:
            llvm::StringRef name() {
                return sym_name();
            }

            FunctionType getType() {
                return getTypeAttr().getValue().cast<FunctionType>();
            }

            BlockArgument getVariable();
            BlockArgument getValue();
            ArrayRef<BlockArgument> getVariableIndices();

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
            friend class OpTrait::FunctionLike<VariableSetterOp>;

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
        ida.add_variable_access %0, %1, %2 { access = affine_map<(d0, d1)[s0] -> (d1, d0 + s0)> } : !ida.instance, !ida.equation, !ida.variable
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        IDAEquation:$equation,
        IDAVariable:$variable,
        AffineMapAttr:$access);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation `,` $variable attr-dict `:` type($instance) `,` type($equation) `,` type($variable)
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

        The function takes three parameters:
          1. IDA instance
          2. current time of the simulation (which will be provided by IDA when calling the residual function).
          3. indices of the scalar equation to be examined.

        Example:

        ```mlir
        ida.residual_function @foo {
            ^bbo(%instance : !ida.instance, %time : f64, %indices : memref<?xi64>) {
                ...
                %diff = ... : f64
                ida.return %diff : f64
            }
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type,
        IndexAttr:$equationRank);

    let results = (outs);

    let regions = (region
        AnyRegion:$bodyRegion // TODO switch to SizedRegion<1
    );

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name, "Type":$timeType, "TypeRange":$variablesTypes, "size_t":$equationRank, "Type":$differenceType), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 3> args;

            args.push_back(timeType);

            for (const auto& variableType : variablesTypes) {
                args.push_back(variableType);
            }

            for (size_t i = 0; i < equationRank; ++i) {
                args.push_back($_builder.getIndexType());
            }

            auto functionType = $_builder.getFunctionType(args, differenceType);
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr, $_builder.getIndexAttr(equationRank));
        }]>
    ];

    let extraClassDeclaration = [{
        public:
            llvm::StringRef name() {
                return sym_name();
            }

            FunctionType getType() {
                return getTypeAttr().getValue().cast<FunctionType>();
            }

            BlockArgument getTime();
            ArrayRef<BlockArgument> getVariables();
            ArrayRef<BlockArgument> getEquationIndices();

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

        The function takes six parameters:
          1. IDA instance.
          2. current time of the simulation (which will be provided by IDA when calling the residual function).
          3. indices of the scalar equation to be examined.
          4. array variable belonging to the IDA instance.
          5. indices of the scalar variable with respect to which
             the equation has to be derived.
          6. the IDA alpha parameter.

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
            ^bbo(%instance : !ida.instance, %time : f64, %equationIndices : memref<?xi64>, %var : !ida.variable, %variableIndices : memref<?xi64>, %alpha : f64) {
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
        TypeAttr:$type,
        IndexAttr:$equationRank,
        IndexAttr:$variableRank);

    let results = (outs);

    let regions = (region
        AnyRegion:$bodyRegion // TODO switch to SizedRegion<1
    );

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name, "Type":$timeType, "TypeRange":$variablesTypes, "size_t":$equationRank, "size_t":$variableRank, "Type":$alphaType, "Type":$differenceType), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 3> args;

            args.push_back(timeType);

            for (const auto& variableType : variablesTypes) {
                args.push_back(variableType);
            }

            for (size_t i = 0; i < equationRank; ++i) {
                args.push_back($_builder.getIndexType());
            }

            for (size_t i = 0; i < variableRank; ++i) {
                args.push_back($_builder.getIndexType());
            }

           args.push_back(alphaType);

            auto functionType = $_builder.getFunctionType(args, differenceType);
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr, $_builder.getIndexAttr(equationRank), $_builder.getIndexAttr(variableRank));
        }]>
    ];

    let extraClassDeclaration = [{
        public:
            llvm::StringRef name() {
                return sym_name();
            }

            BlockArgument getTime();
            ArrayRef<BlockArgument> getVariables();
            ArrayRef<BlockArgument> getEquationIndices();
            ArrayRef<BlockArgument> getVariableIndices();
            BlockArgument getAlpha();

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
     ParentOneOf<["VariableGetterOp", "VariableSetterOp", "ResidualFunctionOp", "JacobianFunctionOp"]>,
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
        %2 = ... : !ida.variable
        ida.add_jacobian %0, %1, %2 { function : "foo" } : !ida.instance, !ida.equation, !ida.variable
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        IDAEquation:$equation,
        IDAVariable:$variable,
        SymbolNameAttr:$function);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation `,` $variable attr-dict `:` type($instance) `,` type($equation) `,` type($variable)
    }];
}

def IDA_InitOp : IDA_Op<"init",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Initialize the IDA environment.";

    let description = [{
        The operation initializes the IDA environment.
        Note that all the equations and variables must have
        already been added to the IDA instance.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        ...
        %1 = ida.init %0 : !ida.instance
        ```
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_StepOp : IDA_Op<"step",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Perform a step within the simulation.";

    let description = [{
        The operation instructs IDA to perform a simulation step.
        If a time step parameter is given, then the simulation
        time will be advanced by that given amount of time.
        If not, an appropriate time step will be determined by
        IDA through internal heuristics and may be different at
        each call.

        Example:
        ```mlir
        %0 = ... : !ida.instance
        %1 = ida.step %0 { step = 0.01 } : !ida.instance
        ```
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_FreeOp : IDA_Op<"free",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Deallocates the IDA data structures and the array variables.";

    let description = [{
        The operation deallocates the IDA data structures and
        the memory reserved for the array variables that have
        been added to IDA.

        Example:
        ```mlir
        %0 = ... : !ida.instance
        %1 = ida.free %0 : !ida.instance
        ```
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_PrintStatisticsOp : IDA_Op<"print_statistics",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Print statistics regarding IDA.";

    let description = [{
        Print to stderr some statistics regarding IDA.

        Example:
        ```mlir
        %0 = ... : !ida.instance
        ida.print_statistics %0 : !ida.instance
        ```
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

#endif // MARCO_DIALECTS_IDA_IDA_TD
