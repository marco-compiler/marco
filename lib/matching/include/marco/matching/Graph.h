#ifndef MARCO_MATCHING_GRAPH_H
#define MARCO_MATCHING_GRAPH_H

#include <llvm/ADT/iterator_range.h>
#include <llvm/ADT/SmallVector.h>
#include <map>

namespace marco::matching
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
      VertexIterator begin(*this, 0, vertices.size());
      VertexIterator end(*this, vertices.size(), vertices.size());

      return llvm::iterator_range<VertexIterator>(begin, end);
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

      EdgeIterator begin(adj.begin(), adj.end());
      EdgeIterator end(adj.end(), adj.end());

      return llvm::iterator_range<EdgeIterator>(begin, end);
    }

    llvm::iterator_range<IncidentEdgeIterator> getIncidentEdges(VertexDescriptor vertex) const
    {
      const auto& incidentEdges = adj.find(vertex)->second;

      IncidentEdgeIterator begin(vertex, incidentEdges.begin(), incidentEdges.end());
      IncidentEdgeIterator end(vertex, incidentEdges.end(), incidentEdges.end());

      return llvm::iterator_range<IncidentEdgeIterator>(begin, end);
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

    VertexIterator(const Graph<VertexProperty, EdgeProperty>& graph, size_t current, size_t end)
            : graph(&graph), current(current), end(end)
    {
    }

    operator bool() const
    {
      return current != end;
    }

    bool operator==(const VertexIterator& it) const
    {
      return graph == it.graph && current == it.current && end == it.end;
    }

    bool operator!=(const VertexIterator& it) const
    {
      return graph != it.graph || current != it.current || end != it.end;
    }

    VertexIterator& operator++()
    {
      current = std::min(current + 1, end);
      return *this;
    }

    VertexIterator operator++(int)
    {
      auto temp = *this;
      current = std::min(current + 1, end);
      return temp;
    }

    value_type operator*()
    {
      return value_type(graph->vertices[current]);
    }

    private:
    const Graph<VertexProperty, EdgeProperty>* graph;
    size_t current;
    size_t end;
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

    IncidentEdgeIterator(VertexDescriptor from, Iterator current, Iterator end)
            : from(std::move(from)), current(current), end(end)
    {
    }

    operator bool() const
    {
      return current != end;
    }

    bool operator==(const IncidentEdgeIterator& it) const
    {
      return from = it.from && current == it.current && end == it.end;
    }

    bool operator!=(const IncidentEdgeIterator& it) const
    {
      return from != it.from || current != it.current || end != it.end;
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

    value_type operator*()
    {
      auto& edge = *current;
      VertexDescriptor to = edge.first;
      auto* property = edge.second;
      return value_type(from, to, property);
    }

    private:
    VertexDescriptor from;
    Iterator current;
    Iterator end;
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

    using Iterator = typename AdjacencyList::const_iterator;

    EdgeIterator(Iterator current, Iterator end)
            : current(std::move(current)), end(std::move(end)), currentEdge(0)
    {
      skipEmptyLists();
    }

    operator bool() const
    {
      if (current == end)
        return false;

      if (current->second.size() == currentEdge)
        return false;

      return true;
    }

    bool operator==(const EdgeIterator& it) const
    {
      return current == it.current && end == it.end && currentEdge == it.currentEdge;
    }

    bool operator!=(const EdgeIterator& it) const
    {
      return current != it.current || end != it.end || currentEdge != it.currentEdge;
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

    value_type operator*()
    {
      auto from = current->first;
      auto& edge = current->second[currentEdge];
      auto to = edge.first;
      auto* ptr = edge.second;
      return value_type(from, to, ptr);
    }

    private:
    void fetchNext()
    {
      if (current == end)
        return;

      if (currentEdge + 1 == current->second.size())
      {
        ++current;
        currentEdge = 0;
        skipEmptyLists();
      }
      else
      {
        ++currentEdge;
      }
    }

    void skipEmptyLists()
    {
      auto shouldProceed = [&]() -> bool {
        if (current == end)
          return false;

        return current->second.empty();
      };

      while (shouldProceed())
        ++current;
    }

    Iterator current;
    Iterator end;
    size_t currentEdge;
  };
}

#endif	// MARCO_MATCHING_GRAPH_H
