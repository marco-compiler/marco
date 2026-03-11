//===- DotEmitter.h - DOT graph emission for load-provenance analysis -----===//
//
//===----------------------------------------------------------------------===//
//
// All visualization logic for emitting GraphViz DOT files from the
// DataRecomputation pass's load-provenance results.  Separated from the
// analysis code so the ~260 lines of formatting do not clutter the core
// analysis in DataRecomputation.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef MARCO_TRANSFORMS_DATARECOMPUTATION_DOTEMITTER_H
#define MARCO_TRANSFORMS_DATARECOMPUTATION_DOTEMITTER_H

#include "marco/Transforms/DataRecomputation/AnalysisState.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace dr {

/// Produce a stable unique node ID from an operation pointer.
inline std::string dotNodeId(mlir::Operation *op) {
  return llvm::formatv("op_{0}", (const void *)op);
}

/// Escape a string for use inside a DOT label.
inline std::string dotEscape(llvm::StringRef s) {
  std::string result;
  result.reserve(s.size());
  for (char c : s) {
    switch (c) {
    case '"':  result += "\\\""; break;
    case '\\': result += "\\\\"; break;
    case '<':  result += "\\<";  break;
    case '>':  result += "\\>";  break;
    case '{':  result += "\\{";  break;
    case '}':  result += "\\}";  break;
    case '|':  result += "\\|";  break;
    case '\n': result += "\\n";  break;
    default:   result += c;      break;
    }
  }
  return result;
}

/// Compact operation label: print without regions, truncate to 120 chars.
inline std::string dotOpLabel(mlir::Operation *op) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  mlir::OpPrintingFlags flags;
  flags.skipRegions();
  op->print(os, flags);
  os.flush();
  if (buf.size() > 120)
    buf = buf.substr(0, 117) + "...";
  return dotEscape(buf);
}

/// Produce a stable unique node ID for a block argument.
inline std::string dotBlockArgId(mlir::BlockArgument arg) {
  return llvm::formatv("barg_{0}_{1}",
      (const void *)arg.getOwner(), arg.getArgNumber());
}

/// Emit the full SSA operand tree rooted at `rootVal` into the DOT graph.
/// Each defining operation becomes a rounded-box node, block arguments become
/// hexagon nodes, and dashed edges show the dataflow.
inline void emitOperandTree(
    mlir::Value rootVal,
    llvm::raw_ostream &out,
    llvm::DenseSet<mlir::Operation *> &emittedNodes,
    llvm::DenseSet<mlir::Value> &emittedBlockArgs,
    llvm::DenseSet<mlir::Operation *> &processedTreeOps) {
  llvm::SmallVector<mlir::Value, 8> worklist;
  worklist.push_back(rootVal);

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();

    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(current)) {
      if (emittedBlockArgs.insert(current).second) {
        std::string id = dotBlockArgId(blockArg);
        std::string typeBuf;
        llvm::raw_string_ostream tos(typeBuf);
        blockArg.getType().print(tos);
        tos.flush();
        std::string label = llvm::formatv("arg#{0} : {1}",
            blockArg.getArgNumber(), typeBuf);
        out << "  \"" << id
            << "\" [shape=hexagon, style=filled, fillcolor=\"#f0e6ff\", label=\""
            << dotEscape(label) << "\"];\n";
      }
      continue;
    }

    mlir::Operation *defOp = current.getDefiningOp();
    if (!defOp) continue;
    if (!processedTreeOps.insert(defOp).second) continue;

    // Emit node only if not already present (as a store or load node).
    if (emittedNodes.insert(defOp).second) {
      out << "  \"" << dotNodeId(defOp)
          << "\" [shape=box, style=\"filled,rounded\", fillcolor=\"#f5f5f5\", label=\""
          << dotOpLabel(defOp) << "\"];\n";
    }

    for (mlir::Value operand : defOp->getOperands()) {
      std::string fromId;
      if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(operand))
        fromId = dotBlockArgId(barg);
      else if (auto *opDef = operand.getDefiningOp())
        fromId = dotNodeId(opDef);
      if (!fromId.empty()) {
        out << "  \"" << fromId << "\" -> \"" << dotNodeId(defOp)
            << "\" [style=dashed, color=\"#888888\"];\n";
      }
      worklist.push_back(operand);
    }
  }
}

/// Emit a GraphViz DOT file visualizing load-provenance analysis results.
inline void emitProvenanceDot(
    llvm::StringRef filePath,
    llvm::SmallVectorImpl<std::pair<mlir::Operation *, StoreSet *>> &singleLoads,
    llvm::SmallVectorImpl<std::pair<mlir::Operation *, StoreSet *>> &multiLoads,
    llvm::SmallVectorImpl<std::pair<mlir::Operation *, StoreSet *>> &leakedLoads,
    llvm::SmallVectorImpl<std::pair<mlir::Operation *, StoreSet *>> &killedLoads) {
  std::error_code ec;
  llvm::raw_fd_ostream out(filePath, ec, llvm::sys::fs::OF_TextWithCRLF);
  if (ec) {
    llvm::errs() << "error: cannot open DOT file '" << filePath
                 << "': " << ec.message() << "\n";
    return;
  }

  out << "digraph load_provenance {\n";
  out << "  rankdir=LR;\n";
  out << "  node [fontname=\"Courier\", fontsize=10];\n";
  out << "  edge [fontname=\"Courier\", fontsize=9];\n\n";

  // Track emitted nodes to avoid duplicate/conflicting node definitions.
  llvm::DenseSet<mlir::Operation *> emittedNodes;
  llvm::SmallVector<mlir::Operation *> allStoreOps;
  bool needsExternalNode = false;

  // Helper: emit a store node if not already emitted.
  auto emitStoreNode = [&](mlir::Operation *storeOp) {
    if (!storeOp) {
      needsExternalNode = true;
      return;
    }
    if (!emittedNodes.insert(storeOp).second)
      return;
    allStoreOps.push_back(storeOp);
    out << "  \"" << dotNodeId(storeOp)
        << "\" [shape=ellipse, style=filled, fillcolor=\"#d4edda\", label=\""
        << dotOpLabel(storeOp) << "\"];\n";
  };

  // --- SINGLE loads: store --> load ---
  out << "  // === SINGLE loads ===\n";
  for (auto &[loadOp, stores] : singleLoads) {
    emittedNodes.insert(loadOp);
    out << "  \"" << dotNodeId(loadOp)
        << "\" [shape=box, style=filled, fillcolor=\"#cce5ff\", label=\""
        << dotOpLabel(loadOp) << "\"];\n";
    for (mlir::Operation *s : *stores) {
      emitStoreNode(s);
      if (s) {
        out << "  \"" << dotNodeId(s) << "\" -> \"" << dotNodeId(loadOp)
            << "\";\n";
      }
    }
  }

  // --- MULTI loads: stores --> diamond --> load ---
  out << "\n  // === MULTI loads ===\n";
  unsigned mergeId = 0;
  for (auto &[loadOp, stores] : multiLoads) {
    emittedNodes.insert(loadOp);
    out << "  \"" << dotNodeId(loadOp)
        << "\" [shape=box, style=filled, fillcolor=\"#cce5ff\", label=\""
        << dotOpLabel(loadOp) << "\"];\n";

    std::string diamondId = llvm::formatv("merge_{0}", mergeId++);
    out << "  \"" << diamondId
        << "\" [shape=diamond, style=filled, fillcolor=\"#fff3cd\", label=\"merge\"];\n";
    out << "  \"" << diamondId << "\" -> \"" << dotNodeId(loadOp) << "\";\n";

    for (mlir::Operation *s : *stores) {
      emitStoreNode(s);
      if (s) {
        out << "  \"" << dotNodeId(s) << "\" -> \"" << diamondId << "\";\n";
      }
    }
  }

  // --- LEAKED loads: stores --> diamond --> load (red) ---
  out << "\n  // === LEAKED loads ===\n";
  for (auto &[loadOp, stores] : leakedLoads) {
    emittedNodes.insert(loadOp);
    out << "  \"" << dotNodeId(loadOp)
        << "\" [shape=box, style=filled, fillcolor=\"#f8d7da\", "
           "color=red, penwidth=2, label=\""
        << dotOpLabel(loadOp) << "\"];\n";

    std::string diamondId = llvm::formatv("merge_{0}", mergeId++);
    out << "  \"" << diamondId
        << "\" [shape=diamond, style=filled, fillcolor=\"#fff3cd\", label=\"merge\"];\n";
    out << "  \"" << diamondId << "\" -> \"" << dotNodeId(loadOp) << "\";\n";

    for (mlir::Operation *s : *stores) {
      if (!s) {
        needsExternalNode = true;
        out << "  external_write -> \"" << diamondId << "\";\n";
      } else {
        emitStoreNode(s);
        out << "  \"" << dotNodeId(s) << "\" -> \"" << diamondId << "\";\n";
      }
    }
  }

  // --- KILLED loads: all reaching stores were killed ---
  out << "\n  // === KILLED loads ===\n";
  for (auto &[loadOp, stores] : killedLoads) {
    emittedNodes.insert(loadOp);
    out << "  \"" << dotNodeId(loadOp)
        << "\" [shape=box, style=filled, fillcolor=\"#e0e0e0\", "
           "color=\"#666666\", penwidth=2, label=\""
        << dotOpLabel(loadOp) << "\"];\n";
  }

  // Emit the shared external write node if needed.
  if (needsExternalNode) {
    out << "\n  external_write [shape=octagon, style=filled, "
           "fillcolor=\"#e2e3e5\", label=\"<external>\"];\n";
  }

  // --- Operand trees: full SSA computation chains for each store ---
  out << "\n  // === Operand Trees ===\n";
  {
    llvm::DenseSet<mlir::Operation *> processedTreeOps;
    llvm::DenseSet<mlir::Value> emittedBlockArgs;
    for (mlir::Operation *storeOp : allStoreOps) {
      auto memStoreOp = mlir::dyn_cast<mlir::memref::StoreOp>(storeOp);
      if (!memStoreOp) continue;
      mlir::Value val = memStoreOp.getValueToStore();
      std::string fromId;
      if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(val))
        fromId = dotBlockArgId(barg);
      else if (auto *opDef = val.getDefiningOp())
        fromId = dotNodeId(opDef);
      if (!fromId.empty()) {
        out << "  \"" << fromId << "\" -> \"" << dotNodeId(storeOp)
            << "\" [style=dashed, color=\"#888888\", label=\"val\"];\n";
      }
      emitOperandTree(val, out, emittedNodes, emittedBlockArgs,
                       processedTreeOps);
    }
  }

  // Legend subgraph.
  out << R"(
  subgraph cluster_legend {
    label="Legend";
    style=dashed;
    fontsize=12;
    legend_store [shape=ellipse, style=filled, fillcolor="#d4edda", label="store"];
    legend_load [shape=box, style=filled, fillcolor="#cce5ff", label="load"];
    legend_leaked [shape=box, style=filled, fillcolor="#f8d7da", color=red, penwidth=2, label="leaked load"];
    legend_killed [shape=box, style=filled, fillcolor="#e0e0e0", color="#666666", penwidth=2, label="killed load"];
    legend_merge [shape=diamond, style=filled, fillcolor="#fff3cd", label="merge"];
    legend_ext [shape=octagon, style=filled, fillcolor="#e2e3e5", label="<external>"];
    legend_compute [shape=box, style="filled,rounded", fillcolor="#f5f5f5", label="computation"];
    legend_arg [shape=hexagon, style=filled, fillcolor="#f0e6ff", label="block arg"];
  }
)";

  out << "}\n";
  llvm::dbgs() << "DRCOMP: Wrote load-provenance DOT graph to " << filePath
               << "\n";
}

} // namespace dr

#endif // MARCO_TRANSFORMS_DATARECOMPUTATION_DOTEMITTER_H
