#ifndef MARCO_MODELING_GRAPHDUMPER_H
#define MARCO_MODELING_GRAPHDUMPER_H

#include "marco/Modeling/Dumpable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <utility>

namespace marco::modeling::internal {
namespace impl {

//==----------------------------------------------------------------------===//
// Example Backend
//
// Except for `GraphType`, all subsequent template parameters will be
// wrapped functors. See `PrinterWrapper` for details.
// The wrapped functors, themselves, behave like functors.
//==----------------------------------------------------------------------===//

// template <class GraphType, class VPrinter, class EPrinter>
// class GraphDumperTestBackend {
// public:
//   using VertexDescriptor = typename GraphType::VertexDescriptor;
//   using EdgeDescriptor = typename GraphType::EdgeDescriptor;
//
//   using VertexProperty = typename GraphType::VertexProperty;
//   using EdgeProperty = typename GraphType::EdgeProperty;
//
//   // These two overloads exist to allow for default behavior when
//   // the GraphDumper is invoked without an edge printer.
//   void dump(llvm::raw_ostream &os, VPrinter &&vp, EPrinter &&ep) const {
//     GraphDumperTestBackend dumper(graph);
//     dumper.dumpImpl(os, std::forward<VPrinter>(vp),
//     std::forward<EPrinter>(ep));
//   }
//
//   void dump(llvm::raw_ostream &os, VPrinter &&vp) const {
//     dumpImpl(os, std::forward<VPrinter>(vp), nullptr);
//   }
//
//   // The main implementation happens here.
//   // Use the graph to build the representation needed for the backend.
//   // Use the `llvm::raw_ostream` object passed to insert into the correct
//   // stream.
//   void dumpImpl(llvm::raw_ostream &os, VPrinter &&vp, EPrinter &&ep) {
//     for (auto eIt = graph->edgesBegin(); eIt != graph->edgesEnd(); eIt++) {
//
//       auto edgeDescriptor = *eIt;
//
//       const auto fromVertexDescriptor = edgeDescriptor.from;
//       const auto toVertexDescriptor = edgeDescriptor.to;
//
//       const auto fromVertexWrapper = *fromVertexDescriptor.value;
//       const auto toVertexWrapper = *toVertexDescriptor.value;
//
//       const auto fromVertex = *(fromVertexWrapper);
//       const auto toVertex = *(toVertexWrapper);
//
//       vp(fromVertex, os);
//
//       if constexpr (ep.isValid()) {
//         ep(***edgeDescriptor.value, os);
//       } else {
//         os << " --> ";
//       }
//       vp(toVertex, os);
//       os << "\n";
//     }
//   }
//
//   GraphDumperTestBackend(GraphType *graph) : graph{graph} {}
//
// private:
//   GraphType *graph;
// }

//===----------------------------------------------------------------------==//
// Printer Wrapper
//-----------------------------------------------------------------------------
// A simple utility wrapper that wraps all passed printer functors.
// Allows for both dynamic and compile-time checking of whether the passed
// functor is valid.
//===----------------------------------------------------------------------==//
template <class Functor>
class PrinterWrapper {
private:
  Functor f;

public:
  constexpr PrinterWrapper(Functor &&f) : f{std::forward<Functor>(f)} {}

  constexpr bool isValid() const { return true; }

  template <class... Args>
  void operator()(Args &&...args) {
    f(std::forward<Args>(args)...);
  }
};

template <>
class PrinterWrapper<std::nullptr_t> {
private:
public:
  constexpr PrinterWrapper(std::nullptr_t) {}

  constexpr bool isValid() const { return false; }

  template <class... Args>
  void operator()(Args &&...args) {
    (void)sizeof...(args);
  }
};

//==----------------------------------------------------------------------===//
// Mermaid Backend
//==----------------------------------------------------------------------===//

template <class GraphType, class VPrinter,
          class EPrinter = std::function<void()>>
struct GraphDumperMermaidBackend {
  using VertexDescriptor = typename GraphType::VertexDescriptor;
  using EdgeDescriptor = typename GraphType::EdgeDescriptor;

  using VertexProperty = typename GraphType::VertexProperty;
  using EdgeProperty = typename GraphType::EdgeProperty;

  GraphDumperMermaidBackend(GraphType *graph) : graph{graph} {}

  void dump(llvm::raw_ostream &os, VPrinter &&vp, EPrinter &&ep) const {
    GraphDumperMermaidBackend dumper(graph);
    dumper.dumpImpl(os, std::forward<VPrinter>(vp), std::forward<EPrinter>(ep));
  }

  void dump(llvm::raw_ostream &os, VPrinter &&vp) const {
    GraphDumperMermaidBackend dumper(graph);
    dumper.dumpImpl(os, std::forward<VPrinter>(vp), nullptr);
  }

  std::string indexToStringIdentifier(int idx) {
    std::string result;
    constexpr int base = 26;

    while (idx >= 0) {
      result = char('A' + (idx % base)) + result;
      idx = (idx / base) - 1;
    }

    return result;
  }

  void dumpImpl(llvm::raw_ostream &os, VPrinter &&vp, EPrinter &&ep) {
    auto mappings = computeMappings();
    outputNodes(os, mappings, std::forward<VPrinter>(vp));

    for (auto eIt = graph->edgesBegin(); eIt != graph->edgesEnd(); eIt++) {

      auto fromVertex = (*eIt).from;
      auto toVertex = (*eIt).to;

      const std::string &identifier = mappings.at(fromVertex);
      const std::string &toIdentifier = mappings.at(toVertex);

      os << identifier << " --> " << toIdentifier << "\n";
    }
  }

  template <class VPrinterT>
  void callVertexPrinter(VPrinterT &&vp, VertexProperty &prop,
                         llvm::raw_ostream &os) {

    if constexpr (!std::is_same_v<VPrinterT, std::nullptr_t>) {
      os << "(\"";
      std::forward<VPrinter>(vp)(prop, os);
      os << "\")";
    }
  }

  void outputNodes(llvm::raw_ostream &os,
                   llvm::DenseMap<VertexDescriptor, std::string> &mappings,
                   VPrinter &&vp) {
    for (auto vIt = graph->verticesBegin(); vIt != graph->verticesEnd();
         vIt++) {
      std::string identifier = mappings.at(*vIt);
      os << identifier;

      VertexProperty &prop = (**(*vIt).value);
      callVertexPrinter<VPrinter>(std::forward<VPrinter>(vp), prop, os);

      os << "\n";
    }
  }

  llvm::DenseMap<VertexDescriptor, std::string> computeMappings() {
    llvm::DenseMap<VertexDescriptor, std::string> result;

    int idx = 0;
    for (auto vIt = graph->verticesBegin(); vIt != graph->verticesEnd();
         vIt++) {
      std::string identifier = indexToStringIdentifier(idx++);
      result[*vIt] = identifier;
    }

    return result;
  }

private:
  GraphType *graph;
};
} // namespace impl

//==----------------------------------------------------------------------===//
// GraphDumper
//
// This is the main type-erased object that handles forwarding the graph
// and printers to the backend.
//==----------------------------------------------------------------------===//
template <template <class GraphType, class... Printers> class Backend =
              impl::GraphDumperMermaidBackend>
struct GraphDumper : public internal::Dumpable {
  void dump(llvm::raw_ostream &os) const override { pimpl->dump(os); }

  template <class GraphType, class... Printers>
  GraphDumper(GraphType *graph, Printers &&...printers) {
    using GraphWrapperImplType =
        GraphWrapperImpl<GraphType, impl::PrinterWrapper<Printers>...>;

    pimpl = std::make_unique<GraphWrapperImplType>(
        graph,
        impl::PrinterWrapper<Printers>(std::forward<Printers>(printers))...);
  }

  struct GraphWrapperInterface {
    virtual ~GraphWrapperInterface() = default;
    virtual void dump(llvm::raw_ostream &os) = 0;
  };

  template <class GraphType, class... Printers>
  struct GraphWrapperImpl : public GraphWrapperInterface {
    ~GraphWrapperImpl() final = default;

    GraphWrapperImpl(GraphType *graph, Printers &&...printers)
        : dumper{graph},
          printers{std::make_tuple(std::forward<Printers>(printers)...)} {}

    void dump(llvm::raw_ostream &os) final {
      dumpImpl(os, std::index_sequence_for<Printers...>{});
    }

    template <std::size_t... Indices>
    void dumpImpl(llvm::raw_ostream &os, std::index_sequence<Indices...>) {
      dumper.dump(os, std::move(std::get<Indices>(printers))...);
    }

    Backend<GraphType, Printers...> dumper;
    std::tuple<Printers...> printers;
  };

  std::unique_ptr<GraphWrapperInterface> pimpl;
};

template <
    template <class, class...> class Backend = impl::GraphDumperMermaidBackend>
GraphDumper() -> GraphDumper<Backend>;

} // namespace marco::modeling::internal

#endif // MARCO_MODELING_GRAPHDUMPER_H
