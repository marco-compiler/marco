#ifndef MARCO_CODEGEN_LOWERING_VARIABLESSYMBOLTABLE_H
#define MARCO_CODEGEN_LOWERING_VARIABLESSYMBOLTABLE_H

#include "marco/Codegen/Lowering/Reference.h"
#include "llvm/ADT/ScopedHashTable.h"
#include <optional>
#include <set>
#include <string>

/// @file VariablesSymbolTable.h
/// @brief Header file for the VariablesSymbolTable class.
///
/// The VariablesSymbolTable class is a wrapper around a llvm::ScopedHashTable
/// and a set containg the table's keys.

namespace marco::codegen::lowering {
/// @class marco::codegen::lowering::VariablesSymbolTable
/// @brief A wrapper around a llvm::ScopedHashTable and a set of keys.
///
/// This class is a wrapper around a llvm::ScopedHashTable and a set.
/// The table's entries relative to a specific scope are removed when the
/// scope is destroyed.
/// The role of the set is storing the keys that have been inserted into the
/// table. When entries are removed from the table, they are not removed from
/// the set, and the class provides a way to get the current entries of the
/// table and a way to get the entries that have been inserted from its
/// creation.
class VariablesSymbolTable {
public:
  /// @class marco::codegen::lowering::VariablesScope
  /// @brief A wrapper around a llvm::ScopedHashTable::ScopeTy.
  ///
  /// This class is a wrapper around a llvm::ScopedHashTable::ScopeTy and
  /// provides a constructor that accepts a VariablesSymbolTable instead of
  /// a llvm::ScopedHashTable.
  class VariablesScope {
  public:
    explicit VariablesScope(VariablesSymbolTable &ht);

  private:
    llvm::ScopedHashTable<llvm::StringRef, Reference>::ScopeTy scope;
  };

  /// Insert an entry into the table.
  void insert(const llvm::StringRef &Key, const Reference &Val);

  /// Return the number of entries with the given key in the table, among the
  /// ones present in the llvm::ScopedHashTable.
  unsigned int count(const llvm::StringRef &Key) const;

  /// If the given key is in the llvm::ScopedHashTable, return its entry.
  /// Otherwise, return a nullopt.
  std::optional<Reference> lookup(const llvm::StringRef &Key) const;

  /// If onlyVisible is false, return all entries inserted in the table from
  /// its creation. Otherwise, return only the entries currently present in
  /// the llvm::ScopedHashTable.
  std::set<std::string> getVariables(bool onlyVisible = true) const;

private:
  llvm::ScopedHashTable<llvm::StringRef, Reference> references;
  std::set<std::string> variablesSet;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_VARIABLESSYMBOLTABLE_H