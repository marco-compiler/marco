#ifndef MARCO_MATCHING_GRAPH_H
#define MARCO_MATCHING_GRAPH_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/ADT/SmallVector.h>
#include <map>

namespace marco::matching::base
{
  namespace detail
  {
    template<typename T>
    struct VertexDescriptorWrapper
    {
      public:
      VertexDescriptorWrapper(T* value) : value(std::move(value))
      {
      }

      bool operator==(const VertexDescriptorWrapper<T>& other) const
      {
        return value == other.value;
      }

      bool operator!=(const VertexDescriptorWrapper<T>& other) const
      {
        return value != other.value;
      }

      bool operator<(const VertexDescriptorWrapper<T>& other) const
      {
        return value < other.value;
      }

      T* value;
    };

    template<typename T, typename VertexDescriptor>
    struct EdgeDescriptorWrapper
    {
      public:
      EdgeDescriptorWrapper(VertexDescriptor from, VertexDescriptor to, T* value) : from(std::move(from)), to(std::move(to)), value(std::move(value))
      {
      }

      bool operator==(const EdgeDescriptorWrapper<T, VertexDescriptor>& other) const
      {
        return from == other.from && to == other.to && value == other.value;
      }

      bool operator!=(const EdgeDescriptorWrapper<T, VertexDescriptor>& other) const
      {
        return from != other.from || to != other.to || value != other.value;
      }

      VertexDescriptor from;
      VertexDescriptor to;
      T* value;
    };

    // TODO: use SFINAE to check that Iterator iterates on values with Descriptor type
    template<typename Descriptor, typename Iterator>
    class FilteredDescriptorIterator
    {
      public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = Descriptor;
      using difference_type = std::ptrdiff_t;
      using pointer = Descriptor*;
      using reference = Descriptor&;

      FilteredDescriptorIterator(Iterator currentIt, Iterator endIt, std::function<bool(Descriptor)> visibilityFn)
              : currentIt(std::move(currentIt)),
                endIt(std::move(endIt)),
                visibilityFn(std::move(visibilityFn))
      {
        if (shouldProceed())
          fetchNext();
      }

      bool operator==(const FilteredDescriptorIterator& it) const
      {
        return currentIt == it.currentIt && endIt == it.endIt;
      }

      bool operator!=(const FilteredDescriptorIterator& it) const
      {
        return currentIt != it.currentIt || endIt != it.endIt;
      }

      FilteredDescriptorIterator& operator++()
      {
        fetchNext();
        return *this;
      }

      FilteredDescriptorIterator operator++(int)
      {
        auto temp = *this;
        fetchNext();
        return temp;
      }

      Descriptor operator*() const
      {
        return *currentIt;
      }

      private:
      bool shouldProceed() const
      {
        if (currentIt == endIt)
          return false;

        auto descriptor = *currentIt;
        return !visibilityFn(descriptor);
      }

      void fetchNext()
      {
        if (currentIt == endIt)
          return;

        do {
          ++currentIt;
        } while (shouldProceed());
      }

      Iterator currentIt;
      Iterator endIt;
      std::function<bool(Descriptor)> visibilityFn;
    };
  }

  template<typename VertexProperty, typename EdgeProperty>
  class Graph
  {
    public:
    using VertexDescriptor = detail::VertexDescriptorWrapper<VertexProperty>;
    using EdgeDescriptor = detail::EdgeDescriptorWrapper<EdgeProperty, VertexDescriptor>;

    private:
    using IncidentEdgesList = std::vector<std::pair<VertexDescriptor, EdgeProperty*>>;
    using AdjacencyList = std::map<VertexDescriptor, IncidentEdgesList>;

    public:
    class VertexIterator;
    class IncidentEdgeIterator;
    class EdgeIterator;

    using FilteredVertexIterator = detail::FilteredDescriptorIterator<VertexDescriptor, VertexIterator>;
    using FilteredIncidentEdgeIterator = detail::FilteredDescriptorIterator<EdgeDescriptor, IncidentEdgeIterator>;
    using FilteredEdgeIterator = detail::FilteredDescriptorIterator<EdgeDescriptor, EdgeIterator>;

    ~Graph()
    {
      for (auto& vertex : vertices)
        delete vertex;

      for (auto& edge : edges)
        delete edge;
    }

    VertexProperty& operator[](VertexDescriptor descriptor)
    {
      return *descriptor.value;
    }

    const VertexProperty& operator[](VertexDescriptor descriptor) const
    {
      return *descriptor.value;
    }

    EdgeProperty& operator[](EdgeDescriptor descriptor)
    {
      return *descriptor.value;
    }

    const EdgeProperty& operator[](EdgeDescriptor descriptor) const
    {
      return *descriptor.value;
    }

    VertexDescriptor addVertex(VertexProperty property)
    {
      auto* ptr = new VertexProperty(std::move(property));
      vertices.push_back(ptr);
      return VertexDescriptor(ptr);
    }

    llvm::iterator_range<VertexIterator> getVertices() const
    {
      VertexIterator begin(vertices.begin());
      VertexIterator end(vertices.end());

      return llvm::iterator_range<VertexIterator>(begin, end);
    }

    llvm::iterator_range<FilteredVertexIterator> getVertices(
            std::function<bool(const VertexProperty&)> visibilityFn) const
    {
      auto filter = [&](const VertexDescriptor& descriptor) -> bool {
          return visibilityFn((*this)[descriptor]);
      };

      auto allVertices = getVertices();

      FilteredVertexIterator begin(allVertices.begin(), allVertices.end(), filter);
      FilteredVertexIterator end(allVertices.end(), allVertices.end(), filter);

      return llvm::iterator_range<FilteredVertexIterator>(begin, end);
    }

    EdgeDescriptor addEdge(VertexDescriptor from, VertexDescriptor to, EdgeProperty property)
    {
      auto* ptr = new EdgeProperty(std::move(property));
      edges.push_back(ptr);

      adj[from].push_back(std::make_pair(to, ptr));
      adj[to].push_back(std::make_pair(from, ptr));

      return EdgeDescriptor(from, to, ptr);
    }

    llvm::iterator_range<EdgeIterator> getEdges() const
    {
      auto verticesDescriptors = getVertices();

      EdgeIterator begin(verticesDescriptors.begin(), verticesDescriptors.end(), adj);
      EdgeIterator end(verticesDescriptors.end(), verticesDescriptors.end(), adj);

      return llvm::iterator_range<EdgeIterator>(begin, end);
    }

    llvm::iterator_range<FilteredEdgeIterator> getEdges(
            std::function<bool(const EdgeProperty&)> visibilityFn) const
    {
      auto filter = [&](const EdgeDescriptor& descriptor) -> bool {
          return visibilityFn((*this)[descriptor]);
      };

      auto allEdges = getEdges();

      FilteredEdgeIterator begin(allEdges.begin(), allEdges.end(), filter);
      FilteredEdgeIterator end(allEdges.end(), allEdges.end(), filter);

      return llvm::iterator_range<FilteredEdgeIterator>(begin, end);
    }

    llvm::iterator_range<IncidentEdgeIterator> getIncidentEdges(VertexDescriptor vertex) const
    {
      const auto& incidentEdges = adj.find(vertex)->second;

      IncidentEdgeIterator begin(vertex, incidentEdges.begin());
      IncidentEdgeIterator end(vertex, incidentEdges.end());

      return llvm::iterator_range<IncidentEdgeIterator>(begin, end);
    }

    llvm::iterator_range<FilteredIncidentEdgeIterator> getIncidentEdges(
            VertexDescriptor vertex,
            std::function<bool(const EdgeProperty&)> visibilityFn) const
    {
      auto filter = [&](const EdgeDescriptor& descriptor) -> bool {
          return visibilityFn((*this)[descriptor]);
      };

      auto allEdges = getIncidentEdges(vertex);

      FilteredIncidentEdgeIterator begin(allEdges.begin(), allEdges.end(), filter);
      FilteredIncidentEdgeIterator end(allEdges.end(), allEdges.end(), filter);

      return llvm::iterator_range<FilteredIncidentEdgeIterator>(begin, end);
    }

    private:
    llvm::SmallVector<VertexProperty*, 3> vertices;
    llvm::SmallVector<EdgeProperty*, 3> edges;
    AdjacencyList adj;
  };

  template<typename VertexProperty, typename EdgeProperty>
  class Graph<VertexProperty, EdgeProperty>::VertexIterator
  {
    public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = VertexDescriptor;
    using difference_type = std::ptrdiff_t;
    using pointer = VertexDescriptor*;
    using reference = VertexDescriptor&;

    using Iterator = typename decltype(vertices)::const_iterator;

    VertexIterator(Iterator current) : current(current)
    {
    }

    bool operator==(const VertexIterator& it) const
    {
      return current == it.current;
    }

    bool operator!=(const VertexIterator& it) const
    {
      return current != it.current;
    }

    VertexIterator& operator++()
    {
      ++current;
      return *this;
    }

    VertexIterator operator++(int)
    {
      auto temp = *this;
      ++current;
      return temp;
    }

    VertexDescriptor operator*() const
    {
      return VertexDescriptor(*current);
    }

    private:
    Iterator current;
  };

  template<typename VertexProperty, typename EdgeProperty>
  class Graph<VertexProperty, EdgeProperty>::IncidentEdgeIterator
  {
    public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = EdgeDescriptor;
    using difference_type = std::ptrdiff_t;
    using pointer = EdgeDescriptor*;
    using reference = EdgeDescriptor&;

    using Iterator = typename IncidentEdgesList::const_iterator;

    IncidentEdgeIterator(VertexDescriptor from, Iterator current)
            : from(std::move(from)), current(current)
    {
    }

    bool operator==(const IncidentEdgeIterator& it) const
    {
      return from = it.from && current == it.current;
    }

    bool operator!=(const IncidentEdgeIterator& it) const
    {
      return from != it.from || current != it.current;
    }

    IncidentEdgeIterator& operator++()
    {
      ++current;
      return *this;
    }

    IncidentEdgeIterator operator++(int)
    {
      auto temp = *this;
      ++current;
      return temp;
    }

    EdgeDescriptor operator*() const
    {
      auto& edge = *current;
      VertexDescriptor to = edge.first;
      auto* property = edge.second;
      return EdgeDescriptor(from, to, property);
    }

    private:
    VertexDescriptor from;
    Iterator current;
  };

  template<typename VertexProperty, typename EdgeProperty>
  class Graph<VertexProperty, EdgeProperty>::EdgeIterator
  {
    public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = EdgeDescriptor;
    using difference_type = std::ptrdiff_t;
    using pointer = EdgeDescriptor*;
    using reference = EdgeDescriptor&;

    EdgeIterator(VertexIterator currentVertexIt, VertexIterator endVertexIt, const AdjacencyList& adj)
            : currentVertexIt(std::move(currentVertexIt)),
              endVertexIt(std::move(endVertexIt)),
              adj(&adj),
              currentEdge(0)
    {
      if (shouldProceed())
        fetchNext();
    }

    bool operator==(const EdgeIterator& it) const
    {
      return currentVertexIt == it.currentVertexIt && currentEdge == it.currentEdge;
    }

    bool operator!=(const EdgeIterator& it) const
    {
      return currentVertexIt != it.currentVertexIt || currentEdge != it.currentEdge;
    }

    EdgeIterator& operator++()
    {
      fetchNext();
      return *this;
    }

    EdgeIterator operator++(int)
    {
      auto temp = *this;
      fetchNext();
      return temp;
    }

    EdgeDescriptor operator*() const
    {
      VertexDescriptor from = *currentVertexIt;
      auto& incidentEdges = adj->find(from)->second;
      auto& edge = incidentEdges[currentEdge];
      auto to = edge.first;
      auto* ptr = edge.second;
      return EdgeDescriptor(from, to, ptr);
    }

    private:
    bool shouldProceed() const
    {
      if (currentVertexIt == endVertexIt)
        return false;

      VertexDescriptor from = *currentVertexIt;
      auto& incidentEdges = adj->find(from)->second;

      if (currentEdge == incidentEdges.size())
        return true;

      auto& edge = incidentEdges[currentEdge];
      return from < edge.first;
    }

    void fetchNext()
    {
      if (currentVertexIt == endVertexIt)
        return;

      do
      {
        VertexDescriptor from = *currentVertexIt;
        auto& incidentEdges = adj->find(from)->second;

        if (currentEdge == incidentEdges.size())
        {
          ++currentVertexIt;
          currentEdge = 0;
        }
        else
        {
          ++currentEdge;
        }
      } while (shouldProceed());
    }

    VertexIterator currentVertexIt;
    VertexIterator endVertexIt;
    const AdjacencyList* adj;
    size_t currentEdge;
  };
}

#endif	// MARCO_MATCHING_GRAPH_H
