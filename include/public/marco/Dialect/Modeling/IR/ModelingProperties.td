#ifndef MARCO_DIALECT_MODELING_IR_MODELINGPROPERTIES_TD
#define MARCO_DIALECT_MODELING_IR_MODELINGPROPERTIES_TD

include "mlir/IR/OpBase.td"

def IndexSetProperty : Property<"::mlir::modeling::IndexSet">
{
    let convertToAttribute = [{
        ::mlir::modeling::getPropertiesAsAttribute($_ctxt, $_storage)
    }];

    let convertFromAttribute = [{
        return ::mlir::modeling::setPropertiesFromAttribute(
            $_storage, $_attr, $_diag);
    }];

    let hashProperty = [{
        ::mlir::modeling::computeHash($_storage);
    }];

    let writeToMlirBytecode = [{
        ::mlir::modeling::writeToMlirBytecode($_writer, $_storage)
    }];

    let readFromMlirBytecode = [{
        if (::mlir::failed(::mlir::modeling::readFromMlirBytecode($_reader, $_storage)))
            return ::mlir::failure();
    }];
}

def IndexSetsListProperty : Property<"::mlir::modeling::IndexSetsList">
{
    let convertToAttribute = [{
        ::mlir::modeling::getPropertiesAsAttribute($_ctxt, $_storage)
    }];

    let convertFromAttribute = [{
        return ::mlir::modeling::setPropertiesFromAttribute(
            $_storage, $_attr, $_diag);
    }];

    let hashProperty = [{
        ::mlir::modeling::computeHash($_storage);
    }];

    let writeToMlirBytecode = [{
        ::mlir::modeling::writeToMlirBytecode($_writer, $_storage)
    }];

    let readFromMlirBytecode = [{
        if (::mlir::failed(::mlir::modeling::readFromMlirBytecode($_reader, $_storage)))
            return ::mlir::failure();
    }];
}

#endif // MARCO_DIALECT_MODELING_IR_MODELINGPROPERTIES_TD
