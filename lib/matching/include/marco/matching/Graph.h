#ifndef MARCO_MATCHING_GRAPH_H
#define MARCO_MATCHING_GRAPH_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/ADT/SmallVector.h>
#include <map>

namespace marco::matching::detail
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
  template<typename Iterator>
  class FilteredIterator
  {
    public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = typename Iterator::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = typename Iterator::pointer;
    using reference = typename Iterator::reference;

    FilteredIterator(Iterator currentIt, Iterator endIt, std::function<bool(value_type)> visibilityFn)
            : currentIt(std::move(currentIt)),
              endIt(std::move(endIt)),
              visibilityFn(std::move(visibilityFn))
    {
      if (shouldProceed())
        fetchNext();
    }

    bool operator==(const FilteredIterator& it) const
    {
      return currentIt == it.currentIt && endIt == it.endIt;
    }

    bool operator!=(const FilteredIterator& it) const
    {
      return currentIt != it.currentIt || endIt != it.endIt;
    }

    FilteredIterator& operator++()
    {
      fetchNext();
      return *this;
    }

    FilteredIterator operator++(int)
    {
      auto temp = *this;
      fetchNext();
      return temp;
    }

    value_type operator*() const
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
    std::function<bool(value_type)> visibilityFn;
  };

  template<typename VertexDescriptor, typename VerticesContainer>
  class VertexIterator
  {
    public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = VertexDescriptor;
    using difference_type = std::ptrdiff_t;
    using pointer = VertexDescriptor*;
    using reference = VertexDescriptor&;

    private:
    using Iterator = typename VerticesContainer::const_iterator;

    public:
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

  template<typename EdgeDescriptor, typename VertexIterator, typename AdjacencyList>
  class EdgeIterator
  {
    public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = EdgeDescriptor;
    using difference_type = std::ptrdiff_t;
    using pointer = EdgeDescriptor*;
    using reference = EdgeDescriptor&;

    using VertexDescriptor = typename VertexIterator::value_type;

    EdgeIterator(VertexIterator currentVertexIt, VertexIterator endVertexIt, const AdjacencyList& adj, bool directed)
            : currentVertexIt(std::move(currentVertexIt)),
              endVertexIt(std::move(endVertexIt)),
              adj(&adj),
              currentEdge(0),
              directed(directed)
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
      auto incidentEdgesIt = adj->find(from);

      if (incidentEdgesIt == adj->end())
        return true;

      auto& incidentEdges = incidentEdgesIt->second;

      if (currentEdge == incidentEdges.size())
        return true;

      auto& edge = incidentEdges[currentEdge];

      if (directed)
        return true;

      return from < edge.first;
    }

    void fetchNext()
    {
      if (currentVertexIt == endVertexIt)
        return;

      do
      {
        VertexDescriptor from = *currentVertexIt;

        auto incidentEdgesIt = adj->find(from);
        bool advanceToNextVertex = incidentEdgesIt == adj->end();

        if (!advanceToNextVertex)
        {
          auto& incidentEdges = incidentEdgesIt->second;
          advanceToNextVertex = currentEdge == incidentEdges.size();
        }

        if (advanceToNextVertex)
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
    bool directed;
  };

  template<typename VertexDescriptor, typename EdgeDescriptor, typename IncidentEdgesList>
  class IncidentEdgeIterator
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
      return from == it.from && current == it.current;
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
  class Graph
  {
    public:
    using VertexDescriptor = VertexDescriptorWrapper<VertexProperty>;
    using EdgeDescriptor = EdgeDescriptorWrapper<EdgeProperty, VertexDescriptor>;

    private:
    using VerticesContainer = llvm::SmallVector<VertexProperty*, 3>;
    using IncidentEdgesList = std::vector<std::pair<VertexDescriptor, EdgeProperty*>>;
    using AdjacencyList = std::map<VertexDescriptor, IncidentEdgesList>;

    public:
    using VertexIterator = detail::VertexIterator<VertexDescriptor, VerticesContainer>;
    using FilteredVertexIterator = detail::FilteredIterator<VertexIterator>;

    using EdgeIterator = detail::EdgeIterator<EdgeDescriptor, VertexIterator, AdjacencyList>;
    using FilteredEdgeIterator = detail::FilteredIterator<EdgeIterator>;

    using IncidentEdgeIterator = detail::IncidentEdgeIterator<VertexDescriptor, EdgeDescriptor, IncidentEdgesList>;
    using FilteredIncidentEdgeIterator = detail::FilteredIterator<IncidentEdgeIterator>;

    Graph(bool directed) : directed(directed)
    {
    }

    virtual ~Graph()
    {
      for (auto* vertex : vertices)
        delete vertex;

      for (auto* edge : edges)
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

    virtual EdgeProperty& operator[](EdgeDescriptor descriptor)
    {
      return *descriptor.value;
    }

    virtual const EdgeProperty& operator[](EdgeDescriptor descriptor) const
    {
      return *descriptor.value;
    }

    VertexDescriptor addVertex(VertexProperty property)
    {
      auto* ptr = new VertexProperty(std::move(property));
      vertices.push_back(ptr);
      VertexDescriptor result(ptr);
      adj.emplace(result, IncidentEdgesList());
      return result;
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
      auto filter = [=](const VertexDescriptor& descriptor) -> bool {
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

      if (!directed)
        adj[to].push_back(std::make_pair(from, ptr));

      return EdgeDescriptor(from, to, ptr);
    }

    llvm::iterator_range<EdgeIterator> getEdges() const
    {
      auto verticesDescriptors = this->getVertices();

      EdgeIterator begin(verticesDescriptors.begin(), verticesDescriptors.end(), adj, directed);
      EdgeIterator end(verticesDescriptors.end(), verticesDescriptors.end(), adj, directed);

      return llvm::iterator_range<EdgeIterator>(begin, end);
    }

    llvm::iterator_range<FilteredEdgeIterator> getEdges(
            std::function<bool(const EdgeProperty&)> visibilityFn) const
    {
      auto filter = [=](const EdgeDescriptor& descriptor) -> bool {
          return visibilityFn((*this)[descriptor]);
      };

      auto allEdges = getEdges();

      FilteredEdgeIterator begin(allEdges.begin(), allEdges.end(), filter);
      FilteredEdgeIterator end(allEdges.end(), allEdges.end(), filter);

      return llvm::iterator_range<FilteredEdgeIterator>(begin, end);
    }

    llvm::iterator_range<IncidentEdgeIterator> getIncidentEdges(VertexDescriptor vertex) const
    {
      auto it = adj.find(vertex);
      assert(it != this->adj.end());
      const auto& incidentEdges = it->second;

      IncidentEdgeIterator begin(vertex, incidentEdges.begin());
      IncidentEdgeIterator end(vertex, incidentEdges.end());

      return llvm::iterator_range<IncidentEdgeIterator>(begin, end);
    }

    llvm::iterator_range<FilteredIncidentEdgeIterator> getIncidentEdges(
            VertexDescriptor vertex,
            std::function<bool(const EdgeProperty&)> visibilityFn) const
    {
      auto filter = [=](const EdgeDescriptor& descriptor) -> bool {
          return visibilityFn((*this)[descriptor]);
      };

      auto allEdges = getIncidentEdges(vertex);

      FilteredIncidentEdgeIterator begin(allEdges.begin(), allEdges.end(), filter);
      FilteredIncidentEdgeIterator end(allEdges.end(), allEdges.end(), filter);

      return llvm::iterator_range<FilteredIncidentEdgeIterator>(begin, end);
    }

    private:
    bool directed;
    VerticesContainer vertices;
    llvm::SmallVector<EdgeProperty*, 3> edges;
    AdjacencyList adj;
  };

  template<typename VertexProperty, typename EdgeProperty>
  class UndirectedGraph : public Graph<VertexProperty, EdgeProperty>
  {
    public:
    using VertexDescriptor = typename Graph<VertexProperty, EdgeProperty>::VertexDescriptor;
    using EdgeDescriptor = typename Graph<VertexProperty, EdgeProperty>::EdgeDescriptor;

    using VertexIterator = typename Graph<VertexProperty, EdgeProperty>::VertexIterator;
    using FilteredVertexIterator = typename Graph<VertexProperty, EdgeProperty>::FilteredVertexIterator;

    using EdgeIterator = typename Graph<VertexProperty, EdgeProperty>::EdgeIterator;
    using FilteredEdgeIterator = typename Graph<VertexProperty, EdgeProperty>::FilteredEdgeIterator;

    using IncidentEdgeIterator = typename Graph<VertexProperty, EdgeProperty>::IncidentEdgeIterator;
    using FilteredIncidentEdgeIterator = typename Graph<VertexProperty, EdgeProperty>::FilteredIncidentEdgeIterator;

    UndirectedGraph() : Graph<VertexProperty, EdgeProperty>(false)
    {
    }
  };

  template<typename VertexProperty, typename EdgeProperty>
  class DirectedGraph : public Graph<VertexProperty, EdgeProperty>
  {
    public:
    using VertexDescriptor = typename Graph<VertexProperty, EdgeProperty>::VertexDescriptor;
    using EdgeDescriptor = typename Graph<VertexProperty, EdgeProperty>::EdgeDescriptor;

    using VertexIterator = typename Graph<VertexProperty, EdgeProperty>::VertexIterator;
    using FilteredVertexIterator = typename Graph<VertexProperty, EdgeProperty>::FilteredVertexIterator;

    using EdgeIterator = typename Graph<VertexProperty, EdgeProperty>::EdgeIterator;
    using FilteredEdgeIterator = typename Graph<VertexProperty, EdgeProperty>::FilteredEdgeIterator;

    using IncidentEdgeIterator = typename Graph<VertexProperty, EdgeProperty>::IncidentEdgeIterator;
    using FilteredIncidentEdgeIterator = typename Graph<VertexProperty, EdgeProperty>::FilteredIncidentEdgeIterator;

    DirectedGraph() : Graph<VertexProperty, EdgeProperty>(true)
    {
    }
  };
}

#endif	// MARCO_MATCHING_GRAPH_H
